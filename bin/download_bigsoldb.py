#!/usr/bin/env python3
import argparse, os, sys, time, hashlib, gzip, shutil, csv
from io import TextIOWrapper
from typing import Optional, List, Dict
import requests

MAIN_HINTS_ANY = {"smiles", "smiles_solute", "smiles_solvent", "logs(mol/l)", "solubility"}
MAIN_REQUIRED_ANY = [{"smiles_solute", "logs(mol/l)"},
                     {"smiles", "solubility"},
                     {"smiles_solvent", "smiles_solute"}]

SOLVENT_HINTS_ANY = {"density", "temperature_k", "solvent"}

def parse_args():
    ap = argparse.ArgumentParser(description="Descarga BigSolDB v2 desde Zenodo.")
    ap.add_argument("--record", required=True, help="Zenodo record id (ej. 15094979)")
    ap.add_argument("--out", required=True, help="Ruta de salida CSV (fichero)")
    ap.add_argument("--kind", choices=["main","solvent","auto"], default="auto",
                    help="Qué CSV descargar: dataset principal, tabla de solventes o autodetect.")
    ap.add_argument("--timeout", type=int, default=60)
    return ap.parse_args()

def zenodo_list_files(record_id: str, timeout: int=60) -> List[Dict]:
    api = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(api, timeout=timeout, headers={"User-Agent": "DrugSolPipeline/1.0"})
    r.raise_for_status()
    data = r.json()
    files = data.get("files") or []
    if not files:
        raise RuntimeError("No hay 'files' en el record de Zenodo.")
    return files

def choose_file(files: List[Dict], kind: str) -> Dict:
    # Primero, si el usuario pidió kind explícito por nombre, intentamos por nombre
    def is_csv(f): return f.get("key","").lower().endswith((".csv",".csv.gz"))
    csvs = [f for f in files if is_csv(f)]
    if not csvs:
        raise RuntimeError("No hay CSV en el record de Zenodo.")

    if kind in ("main","solvent"):
        # heurística por nombre
        pos = [f for f in csvs if kind in f.get("key","").lower()]
        if pos:
            return pos[0]

    # Si auto o no se encontró por nombre, elegimos luego de mirar cabecera
    # Estrategia: probamos HEAD de las primeras ~100 KB para ver cabecera
    ranked = []
    for f in csvs:
        key = f.get("key","")
        url = f.get("links",{}).get("content") or f.get("links",{}).get("self")
        if not url: 
            continue
        kind_guess = sniff_kind(url, key)
        score = 0
        if kind_guess == "main": score = 2
        elif kind_guess == "solvent": score = 1
        # Penaliza densidad si usuario pidió main, y viceversa
        if kind == "main" and kind_guess != "main": score -= 1
        if kind == "solvent" and kind_guess != "solvent": score -= 1
        # desempate por tamaño si está disponible
        size = int(f.get("size") or 0)
        ranked.append((score, size, f))
    if not ranked:
        # fallback: por nombre que contenga 'bigsoldb'
        by_name = [f for f in csvs if "bigsoldb" in f.get("key","").lower()]
        if by_name:
            return by_name[0]
        return csvs[0]
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked[0][2]

def sniff_kind(url: str, name: str) -> str:
    """Descarga unos bytes y mira la cabecera CSV para decidir si es main o solvent."""
    try:
        with requests.get(url, stream=True, timeout=30, headers={"User-Agent":"DrugSolPipeline/1.0"}) as r:
            r.raise_for_status()
            # lee hasta ~128KB
            content = r.raw.read(128*1024)
    except Exception:
        # fallback por nombre
        lname = name.lower()
        if any(t in lname for t in ("density","solvent")):
            return "solvent"
        return "main"

    # si es gz
    if name.lower().endswith(".gz"):
        try:
            import io, gzip as gz
            content = gz.decompress(content)
        except Exception:
            pass

    # parsear cabeceras
    try:
        import io
        text = content.decode("utf-8", errors="ignore")
        reader = csv.reader(text.splitlines())
        header = [h.strip() for h in next(reader)]
        lower = [h.lower() for h in header]
        # reglas
        if _looks_main(lower): return "main"
        if _looks_solvent(lower): return "solvent"
    except Exception:
        pass

    # fallback por nombre
    lname = name.lower()
    if any(t in lname for t in ("density","solvent_properties")):
        return "solvent"
    return "main"

def _looks_main(lower_cols: List[str]) -> bool:
    s = set(lower_cols)
    if s & MAIN_HINTS_ANY:
        # además, comprueba algún conjunto requerido
        for req in MAIN_REQUIRED_ANY:
            if s & {x.lower() for x in req}:
                return True
    # nombres típicos v2
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
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk: f.write(chunk)
            os.replace(tmp, out_path)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return
        if tries >= max_retries:
            raise RuntimeError("Fallo descargando tras reintentos.")
        time.sleep(2)

def _verify_md5(path: str, expected: str):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    got = h.hexdigest()
    if got != expected:
        raise RuntimeError(f"MD5 no coincide: got {got}, expected {expected}")

def main():
    args = parse_args()
    files = zenodo_list_files(args.record, timeout=args.timeout)
    chosen = choose_file(files, args.kind)

    url = chosen.get("links",{}).get("content") or chosen.get("links",{}).get("self")
    if not url:
        raise RuntimeError("No hay URL de descarga para el archivo elegido.")
    checksum = chosen.get("checksum")
    md5_expected = checksum.split(":",1)[1] if checksum and checksum.startswith("md5:") else None

    tmp = args.out + ".part"
    _download_stream(url, tmp, timeout=args.timeout, headers={"User-Agent":"DrugSolPipeline/1.0"})
    if md5_expected:
        _verify_md5(tmp, md5_expected)

    # descomprime si es .gz
    if chosen.get("key","").lower().endswith(".gz") or tmp.lower().endswith(".gz"):
        import gzip
        with gzip.open(tmp,"rb") as src, open(args.out,"wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(tmp)
    else:
        os.replace(tmp, args.out)

if __name__ == "__main__":
    main()
