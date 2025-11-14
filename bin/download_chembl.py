#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from typing import Dict, Any, List, Optional

import requests
import pandas as pd

BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

# Pedimos exactamente los campos necesarios para construir las 4 columnas
FIELDS = ",".join([
    "molecule_structures",      # -> canonical_smiles
    "max_phase",
    "therapeutic_flag",
    "atc_classifications",
    "molecule_type"             # opcional: por si filtras a small molecules
])

def fetch_page(
    offset: int = 0,
    limit: int = 1000,
    only_approved: bool = True,
    small_molecules: bool = True,
    require_smiles: bool = True,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    sess = session or requests.Session()
    params: Dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "fields": FIELDS,
    }
    if only_approved:
        params["max_phase"] = 4
    if small_molecules:
        params["molecule_type"] = "Small molecule"
    if require_smiles:
        # Solo moléculas con estructura (para tener canonical_smiles)
        params["molecule_structures__isnull"] = "false"

    r = sess.get(BASE, params=params, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser(description="Descarga fármacos de ChEMBL y exporta a CSV con 4 columnas.")
    ap.add_argument("--out", required=True, help="Ruta de salida CSV")
    ap.add_argument("--max", type=int, default=3000, help="Máximo de registros a traer")
    ap.add_argument("--pagesize", type=int, default=1000, help="Tamaño de página (<=1000 recomendado)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Pausa entre páginas (seg)")
    ap.add_argument("--only-approved", action="store_true", default=True,
                    help="Solo fármacos aprobados (max_phase=4)")
    ap.add_argument("--small-molecules", action="store_true", default=True,
                    help="Restringir a small molecules")
    ap.add_argument("--require-smiles", action="store_true", default=True,
                    help="Exigir que tenga canonical_smiles")
    args = ap.parse_args()

    sess = requests.Session()
    rows: List[Dict[str, Any]] = []
    got, offset = 0, 0

    while got < args.max:
        data = fetch_page(
            offset=offset,
            limit=args.pagesize,
            only_approved=args.only_approved,
            small_molecules=args.small_molecules,
            require_smiles=args.require_smiles,
            session=sess,
        )
        mols = data.get("molecules", [])
        if not mols:
            break

        for m in mols:
            ms = (m.get("molecule_structures") or {})
            smiles = ms.get("canonical_smiles")

            # Construimos SOLO las 4 columnas requeridas
            rows.append({
                "canonical_smiles": smiles if smiles else "",
                "max_phase": m.get("max_phase"),
                "therapeutic_flag": m.get("therapeutic_flag"),
                "atc_classifications": "|".join(m.get("atc_classifications") or []),
            })

            got += 1
            if got >= args.max:
                break

        offset += args.pagesize
        time.sleep(args.sleep)

    # DataFrame limitado a las 4 columnas y en el orden exacto
    df = pd.DataFrame(rows, columns=[
        "canonical_smiles", "max_phase", "therapeutic_flag", "atc_classifications"
    ])

    # Si se exige smiles, filtramos por no nulos/no vacíos
    if args.require_smiles:
        df = df[df["canonical_smiles"].astype(str).str.len() > 0]

    # Exporta con cabecera EXACTA y sin index
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[ChEMBL] OK: {len(df)} filas → {args.out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ChEMBL] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
