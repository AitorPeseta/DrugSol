#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
    download_bigsoldb.py
========================================================================================
    Downloads BigSolDB v2 dataset from Zenodo repository.
    
    BigSolDB is the largest open-source aqueous solubility database, containing
    experimental measurements for drug-like compounds with standardized metadata.
    
    Features:
    - Automatic CSV detection from Zenodo record
    - Content-based file type identification (main vs solvent tables)
    - MD5 checksum verification for data integrity
    - GZIP decompression support
    - CSV normalization (UTF-8 encoding, Unix line endings)
    - Retry logic for robust downloads

    Arguments:
        --record    Zenodo record ID (e.g., 15094979)
        --out       Output CSV file path
        --kind      File type: 'main', 'solvent', or 'auto' (default: auto)
        --timeout   Download timeout in seconds (default: 60)
        --normalize Normalize CSV after download (recommended)
    
    Usage:
        python fetch_bigsoldb.py --record 15094979 --out bigsoldb

    Output:
        bigsoldb.csv   
        
    Reference:
        https://zenodo.org/records/15094979
----------------------------------------------------------------------------------------
"""

import argparse
import csv
import gzip
import hashlib
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

# ======================================================================================
#     CONSTANTS
# ======================================================================================

# Column hints for identifying the main solubility dataset
MAIN_HINTS_ANY = {"smiles", "smiles_solute", "smiles_solvent", "logs(mol/l)", "solubility"}

# Required column combinations for main dataset validation
MAIN_REQUIRED_ANY = [
    {"smiles_solute", "logs(mol/l)"},
    {"smiles", "solubility"},
    {"smiles_solvent", "smiles_solute"}
]

# Column hints for identifying solvent properties table
SOLVENT_HINTS_ANY = {"density", "temperature_k", "solvent"}

# Download configuration
MAX_RETRIES = 3
CHUNK_SIZE = 1 << 20  # 1 MB chunks
SNIFF_SIZE = 128 * 1024  # 128 KB for header sniffing

# ======================================================================================
#     ARGUMENT PARSING
# ======================================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Download BigSolDB v2 from Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download main dataset with normalization
    python fetch_bigsoldb.py --record 15094979 --out bigsoldb.csv --normalize
    
    # Download solvent properties table
    python fetch_bigsoldb.py --record 15094979 --out solvents.csv --kind solvent
        """
    )
    ap.add_argument(
        "--record", 
        required=True, 
        help="Zenodo record ID (e.g., 15094979)"
    )
    ap.add_argument(
        "--out", 
        required=True, 
        help="Output CSV path"
    )
    ap.add_argument(
        "--kind", 
        choices=["main", "solvent", "auto"], 
        default="auto",
        help="Which CSV to download: main dataset, solvent table, or autodetect (default: auto)"
    )
    ap.add_argument(
        "--timeout", 
        type=int, 
        default=60,
        help="Request timeout in seconds (default: 60)"
    )
    ap.add_argument(
        "--normalize", 
        action="store_true",
        help="Normalize CSV (UTF-8, Unix line endings) after download"
    )
    return ap.parse_args()

# ======================================================================================
#     ZENODO API FUNCTIONS
# ======================================================================================

def zenodo_list_files(record_id: str, timeout: int = 60) -> List[Dict]:
    """
    Fetch file list from Zenodo record via API.
    
    Args:
        record_id: Zenodo record identifier
        timeout: Request timeout in seconds
        
    Returns:
        List of file metadata dictionaries
        
    Raises:
        RuntimeError: If no files found in record
    """
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    response = requests.get(
        api_url, 
        timeout=timeout, 
        headers={"User-Agent": "DrugSolPipeline/1.0"}
    )
    response.raise_for_status()
    
    data = response.json()
    files = data.get("files") or []
    
    if not files:
        raise RuntimeError(f"No files found in Zenodo record {record_id}")
    
    return files

# ======================================================================================
#     FILE SELECTION LOGIC
# ======================================================================================

def choose_file(files: List[Dict], kind: str) -> Dict:
    """
    Select the appropriate CSV file from available files.
    
    Uses a combination of filename matching and content sniffing to identify
    the correct file (main solubility data vs solvent properties).
    
    Args:
        files: List of file metadata from Zenodo
        kind: Desired file type ('main', 'solvent', or 'auto')
        
    Returns:
        Selected file metadata dictionary
        
    Raises:
        RuntimeError: If no CSV files found
    """
    # Filter to CSV files only
    def is_csv(f): 
        return f.get("key", "").lower().endswith((".csv", ".csv.gz"))
    
    csvs = [f for f in files if is_csv(f)]
    if not csvs:
        raise RuntimeError("No CSV files found in Zenodo record")

    # If specific kind requested, try filename matching first
    if kind in ("main", "solvent"):
        matches = [f for f in csvs if kind in f.get("key", "").lower()]
        if matches:
            return matches[0]

    # Content-based ranking using header sniffing
    ranked = []
    for f in csvs:
        key = f.get("key", "")
        url = f.get("links", {}).get("content") or f.get("links", {}).get("self")
        if not url:
            continue
        
        # Sniff content to determine file type
        kind_guess = _sniff_kind(url, key)
        score = 0
        
        if kind_guess == "main":
            score = 2
        elif kind_guess == "solvent":
            score = 1
        
        # Penalize mismatches with requested kind
        if kind == "main" and kind_guess != "main":
            score -= 1
        if kind == "solvent" and kind_guess != "solvent":
            score -= 1
        
        # Tie-breaker: larger files typically contain more data
        size = int(f.get("size") or 0)
        ranked.append((score, size, f))

    if not ranked:
        # Fallback: match by name containing 'bigsoldb'
        by_name = [f for f in csvs if "bigsoldb" in f.get("key", "").lower()]
        if by_name:
            return by_name[0]
        return csvs[0]
    
    # Sort by score (desc) then size (desc)
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked[0][2]


def _sniff_kind(url: str, name: str) -> str:
    """
    Determine file type by sniffing CSV headers.
    
    Downloads first ~128KB to parse headers and apply heuristic rules
    to identify whether file contains main solubility data or solvent properties.
    
    Args:
        url: Download URL for the file
        name: Filename for fallback matching
        
    Returns:
        'main' or 'solvent'
    """
    try:
        with requests.get(
            url, 
            stream=True, 
            timeout=30, 
            headers={"User-Agent": "DrugSolPipeline/1.0"}
        ) as r:
            r.raise_for_status()
            content = r.raw.read(SNIFF_SIZE)
    except Exception:
        # Fallback to filename-based detection
        return _guess_kind_from_name(name)

    # Handle GZIP compression
    if name.lower().endswith(".gz"):
        try:
            content = gzip.decompress(content)
        except Exception:
            pass

    # Parse headers and apply heuristics
    try:
        text = content.decode("utf-8", errors="ignore")
        reader = csv.reader(text.splitlines())
        header = [h.strip() for h in next(reader)]
        lower_cols = [h.lower() for h in header]
        
        if _looks_main(lower_cols):
            return "main"
        if _looks_solvent(lower_cols):
            return "solvent"
    except Exception:
        pass

    return _guess_kind_from_name(name)


def _guess_kind_from_name(name: str) -> str:
    """Fallback: guess file type from filename."""
    lname = name.lower()
    if any(t in lname for t in ("density", "solvent_properties")):
        return "solvent"
    return "main"


def _looks_main(lower_cols: List[str]) -> bool:
    """Check if columns suggest main solubility dataset."""
    cols_set = set(lower_cols)
    
    # Check for hint columns
    if cols_set & MAIN_HINTS_ANY:
        for req in MAIN_REQUIRED_ANY:
            if cols_set & {x.lower() for x in req}:
                return True
    
    # Typical BigSolDB v2 column names
    if "smiles_solute" in cols_set or "logs(mol/l)" in cols_set:
        return True
    
    return False


def _looks_solvent(lower_cols: List[str]) -> bool:
    """Check if columns suggest solvent properties table."""
    cols_set = set(lower_cols)
    
    # Solvent table: has solvent hints but no SMILES columns
    if cols_set & SOLVENT_HINTS_ANY:
        if "smiles" not in cols_set and "smiles_solute" not in cols_set:
            return True
    
    return False

# ======================================================================================
#     DOWNLOAD FUNCTIONS
# ======================================================================================

def download_file(url: str, out_path: str, timeout: int = 60) -> None:
    """
    Download file with retry logic.
    
    Args:
        url: Download URL
        out_path: Output file path
        timeout: Request timeout in seconds
        
    Raises:
        RuntimeError: If download fails after max retries
    """
    headers = {"User-Agent": "DrugSolPipeline/1.0"}
    tmp_path = out_path + ".tmp"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                
                # Atomic move to final location
                os.replace(tmp_path, out_path)
                
                # Verify file exists and is non-empty
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return
                    
        except Exception as e:
            print(f"[WARNING] Download attempt {attempt}/{MAX_RETRIES} failed: {e}")
        
        if attempt < MAX_RETRIES:
            time.sleep(2)
    
    raise RuntimeError(f"Failed to download after {MAX_RETRIES} attempts")


def verify_md5(path: str, expected: str) -> None:
    """
    Verify file MD5 checksum.
    
    Args:
        path: File path to verify
        expected: Expected MD5 hash
        
    Raises:
        RuntimeError: If checksum doesn't match
    """
    md5_hash = hashlib.md5()
    
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            md5_hash.update(chunk)
    
    actual = md5_hash.hexdigest()
    
    if actual != expected:
        raise RuntimeError(f"MD5 mismatch: got {actual}, expected {expected}")
    
    print(f"[BigSolDB] MD5 checksum verified: {actual}")


def decompress_gzip(src_path: str, dst_path: str) -> None:
    """
    Decompress GZIP file.
    
    Args:
        src_path: Compressed file path
        dst_path: Decompressed output path
    """
    print("[BigSolDB] Decompressing GZIP...")
    
    with gzip.open(src_path, "rb") as src:
        with open(dst_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    
    os.remove(src_path)

# ======================================================================================
#     CSV NORMALIZATION
# ======================================================================================

def normalize_csv(filepath: str) -> None:
    """
    Normalize CSV file encoding and line endings.
    
    Ensures consistent format:
    - UTF-8 encoding
    - Unix line endings (LF)
    - Standard CSV formatting
    
    Args:
        filepath: Path to CSV file (modified in-place)
    """
    print(f"[BigSolDB] Normalizing {filepath}...")
    
    try:
        # Pandas handles various encodings robustly
        df = pd.read_csv(filepath)
        
        # Rewrite with standardized format
        df.to_csv(
            filepath, 
            index=False, 
            encoding="utf-8", 
            lineterminator="\n"
        )
        
        print(f"[BigSolDB] Normalization complete. Rows: {len(df):,}")
        
    except Exception as e:
        print(f"[ERROR] Failed to normalize CSV: {e}")
        sys.exit(1)

# ======================================================================================
#     MAIN ENTRY POINT
# ======================================================================================

def main() -> None:
    """Main entry point for BigSolDB download."""
    args = parse_args()
    
    # Fetch file list from Zenodo
    print(f"[BigSolDB] Fetching file list from Zenodo record {args.record}...")
    files = zenodo_list_files(args.record, timeout=args.timeout)
    
    # Select appropriate file
    chosen = choose_file(files, args.kind)
    filename = chosen.get("key", "unknown")
    print(f"[BigSolDB] Selected file: {filename}")
    
    # Get download URL
    url = chosen.get("links", {}).get("content") or chosen.get("links", {}).get("self")
    if not url:
        raise RuntimeError("No download URL found for selected file")
    
    # Extract MD5 checksum if available
    checksum = chosen.get("checksum")
    md5_expected = None
    if checksum and checksum.startswith("md5:"):
        md5_expected = checksum.split(":", 1)[1]
    
    # Download file
    tmp_path = args.out + ".part"
    print(f"[BigSolDB] Downloading from Zenodo...")
    download_file(url, tmp_path, timeout=args.timeout)
    
    # Verify checksum
    if md5_expected:
        verify_md5(tmp_path, md5_expected)
    
    # Decompress if needed
    if filename.lower().endswith(".gz"):
        decompress_gzip(tmp_path, args.out)
    else:
        os.replace(tmp_path, args.out)
    
    # Normalize if requested
    if args.normalize:
        normalize_csv(args.out)
    
    print(f"[BigSolDB] Download complete: {args.out}")


if __name__ == "__main__":
    main()
