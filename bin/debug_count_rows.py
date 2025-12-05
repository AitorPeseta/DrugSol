#!/usr/bin/env python3
"""
debug_count_rows.py
-------------------
Script de diagnóstico rápido para auditar archivos Parquet/CSV.
Comprueba duplicados y consistencia de IDs.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

def analyze_file(path):
    p = Path(path)
    print(f"\n--- Analizando: {p.name} ---")
    
    try:
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
        else:
            print(f"[SKIP] Formato no soportado: {p.suffix}")
            return
    except Exception as e:
        print(f"[ERROR] No se pudo leer: {e}")
        return

    n_rows = len(df)
    print(f"  > Total Filas:      {n_rows}")
    
    # Chequeo de ID
    if "row_uid" in df.columns:
        n_ids = df["row_uid"].nunique()
        print(f"  > row_uid únicos:   {n_ids}")
        if n_rows != n_ids:
            print(f"  ⚠️  ALERTA: Hay {n_rows - n_ids} duplicados de ID!")
            # Mostrar ejemplos
            dups = df[df.duplicated("row_uid", keep=False)].sort_values("row_uid")
            print(f"      Ejemplos: {dups['row_uid'].head(2).tolist()}")
    else:
        print("  [INFO] No hay columna 'row_uid'.")

    # Chequeo de SMILES
    smiles_col = None
    for c in ["smiles_neutral", "smiles", "smiles_original"]:
        if c in df.columns:
            smiles_col = c
            break
            
    if smiles_col:
        n_smi = df[smiles_col].nunique()
        print(f"  > SMILES únicos:    {n_smi} (Columna: {smiles_col})")
        if n_rows != n_smi:
            print(f"  ℹ️  Nota: Hay {n_rows - n_smi} repeticiones de estructura (mismas moléculas, distintos IDs/condiciones).")
    
    # Chequeo de Nulos
    if "logS" in df.columns:
        n_na = df["logS"].isna().sum()
        if n_na > 0:
            print(f"  ⚠️  ALERTA: Hay {n_na} valores nulos en logS.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Archivos a analizar")
    args = parser.parse_args()

    for f in args.files:
        analyze_file(f)

if __name__ == "__main__":
    main()