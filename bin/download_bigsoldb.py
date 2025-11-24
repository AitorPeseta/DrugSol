#!/usr/bin/env python3
import argparse
import os
import sys
import time
import hashlib
import shutil
import csv
import requests
import pandas as pd  # Added for robust CSV normalization
from typing import Optional, List, Dict

# --- CONSTANTS ---
MAIN_HINTS_ANY = {"smiles", "smiles_solute", "smiles_solvent", "logs(mol/l)", "solubility"}
MAIN_REQUIRED_ANY = [{"smiles_solute", "logs(mol/l)"},
                     {"smiles", "solubility"},
                     {"smiles_solvent", "smiles_solute"}]

SOLVENT_HINTS_ANY = {"density", "temperature_k", "solvent"}

def parse_args():
    ap = argparse.ArgumentParser(description="Download BigSolDB v2 from Zenodo.")
    ap.add_argument("--record", required=True, help="Zenodo record ID (e.g., 15094979)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--kind", choices=["main", "solvent", "auto"], default="auto",
                    help="Which CSV to download: main dataset, solvent table, or autodetect.")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--normalize", action="store_true", 
                    help="Normalize CSV (UTF-8, Linux line endings) after download.")
    return ap.parse_args()

def zenodo_list_files(record_id: str, timeout: int=60) -> List[Dict]:
    api = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(api, timeout=timeout, headers={"User-Agent": "DrugSolPipeline/1.0"})
    r.raise_for_status()
    data = r.json()
    files = data.get("files") or []
    if not files:
        raise RuntimeError("No 'files' found in Zenodo record.")
    return files

def choose_file(files: List[Dict], kind: str) -> Dict:
    # Helper: check if file is likely a CSV
    def is_csv(f): return f.get("key", "").lower().endswith((".csv", ".csv.gz"))
    
    csvs = [f for f in files if is_csv(f)]
    if not csvs:
        raise RuntimeError("No CSV files found in Zenodo record.")

    # If specific kind requested, try finding by name first
    if kind in ("main", "solvent"):
        pos = [f for f in csvs if kind in f.get("key", "").lower()]
        if pos:
            return pos[0]

    # Fallback / Auto: Sniff content headers to decide
    # Strategy: Check HEAD of the first ~128KB to parse headers
    ranked = []
    for f in csvs:
        key = f.get("key", "")
        url = f.get("links", {}).get("content") or f.get("links", {}).get("self")
        if not url: 
            continue
            
        kind_guess = sniff_kind(url, key)
        score = 0
        
        if kind_guess == "main": score = 2
        elif kind_guess == "solvent": score = 1
        
        # Penalize if guess contradicts user request
        if kind == "main" and kind_guess != "main": score -= 1
        if kind == "solvent" and kind_guess != "solvent": score -= 1
        
        # Tie-breaker: larger file size usually implies better data
        size = int(f.get("size") or 0)
        ranked.append((score, size, f))

    if not ranked:
        # Fallback: match name containing 'bigsoldb'
        by_name = [f for f in csvs if "bigsoldb" in f.get("key", "").lower()]
        if by_name:
            return by_name[0]
        return csvs[0]
        
    # Sort by score (desc) then size (desc)
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked[0][2]

def sniff_kind(url: str, name: str) -> str:
    """Downloads bytes to sniff CSV headers and decide if it's main or solvent."""
    try:
        with requests.get(url, stream=True, timeout=30, headers={"User-Agent": "DrugSolPipeline/1.0"}) as r:
            r.raise_for_status()
            # Read up to ~128KB
            content = r.raw.read(128*1024)
    except Exception:
        # Fallback: check name
        lname = name.lower()
        if any(t in lname for t in ("density", "solvent")):
            return "solvent"
        return "main"

    # Handle GZIP
    if name.lower().endswith(".gz"):
        try:
            import gzip as gz
            content = gz.decompress(content)
        except Exception:
            pass

    # Parse headers
    try:
        text = content.decode("utf-8", errors="ignore")
        reader = csv.reader(text.splitlines())
        header = [h.strip() for h in next(reader)]
        lower = [h.lower() for h in header]
        
        # Apply heuristic rules
        if _looks_main(lower): return "main"
        if _looks_solvent(lower): return "solvent"
    except Exception:
        pass

    # Fallback by name
    lname = name.lower()
    if any(t in lname for t in ("density", "solvent_properties")):
        return "solvent"
    return "main"

def _looks_main(lower_cols: List[str]) -> bool:
    s = set(lower_cols)
    if s & MAIN_HINTS_ANY:
        # Check for required sets
        for req in MAIN_REQUIRED_ANY:
            if s & {x.lower() for x in req}:
                return True
    # Typical V2 names
    if "smiles_solute" in s or "logs(mol/l)" in s:
        return True
    return False

def _looks_solvent(lower_cols: List[str]) -> bool:
    s = set(lower_cols)
    if s & SOLVENT_HINTS_ANY and "smiles" not in s and "smiles_solute" not in s:
        return True
    return False

def _download_stream(url: str, out_path: str, timeout: int=60, headers: Optional[dict]=None, max_retries: int=3):
    tries = 0
    while True:
        tries += 1
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                tmp = out_path + ".tmp"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk: f.write(chunk)
                os.replace(tmp, out_path)
            
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return
        except Exception as e:
            print(f"[WARNING] Download attempt {tries} failed: {e}")
        
        if tries >= max_retries:
            raise RuntimeError("Failed to download after multiple retries.")
        time.sleep(2)

def _verify_md5(path: str, expected: str):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    got = h.hexdigest()
    if got != expected:
        raise RuntimeError(f"MD5 mismatch: got {got}, expected {expected}")

def normalize_csv(filepath: str):
    """
    Reads the CSV and rewrites it forcing UTF-8 encoding and UNIX line endings.
    Uses Pandas for robustness.
    """
    print(f"[BigSolDB] Normalizing {filepath} ...")
    try:
        # Pandas handles encoding and line terminators robustly
        df = pd.read_csv(filepath)
        df.to_csv(filepath, index=False, encoding="utf-8", lineterminator="\n")
        print(f"[BigSolDB] Normalization successful. Rows: {len(df)}")
    except Exception as e:
        print(f"[ERROR] Failed to normalize CSV: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    files = zenodo_list_files(args.record, timeout=args.timeout)
    chosen = choose_file(files, args.kind)

    url = chosen.get("links", {}).get("content") or chosen.get("links", {}).get("self")
    if not url:
        raise RuntimeError("No download URL found for the chosen file.")
        
    checksum = chosen.get("checksum")
    md5_expected = checksum.split(":", 1)[1] if checksum and checksum.startswith("md5:") else None

    tmp = args.out + ".part"
    print(f"[BigSolDB] Downloading from {url} to {args.out}...")
    _download_stream(url, tmp, timeout=args.timeout, headers={"User-Agent": "DrugSolPipeline/1.0"})
    
    if md5_expected:
        _verify_md5(tmp, md5_expected)

    # Decompress if .gz
    if chosen.get("key", "").lower().endswith(".gz") or tmp.lower().endswith(".gz"):
        print("[BigSolDB] Decompressing GZIP...")
        import gzip
        with gzip.open(tmp, "rb") as src, open(args.out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(tmp)
    else:
        os.replace(tmp, args.out)

    # Normalize if requested
    if args.normalize:
        normalize_csv(args.out)

if __name__ == "__main__":
    main()