#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dropnan_rows.py (CORREGIDO)
---------------------------
Limpieza de datos inteligente.
- Nunca borra filas solo porque falte un descriptor de Mordred/RDKit.
- Solo borra si falta la INFORMACIÓN CRÍTICA (IDs o Target).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def coerce_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(str_cols) > 0:
        df[str_cols] = df[str_cols].replace(r'^\s*$', np.nan, regex=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--mode", choices=["train", "test"], required=True)
    ap.add_argument("--subset", default="all", help="all | features_only")
    ap.add_argument("--save_csv", action="store_true")
    
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] File not found: {args.input}")

    print(f"[Clean] Loading {args.input}...")
    df = read_any(args.input)
    n_original = len(df)

    # Standardize NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = coerce_empty_strings(df)

    # --- LÓGICA CORREGIDA ---
    
    # Identificar columnas críticas (IDs)
    id_keywords = ['smiles_neutral', 'row_uid']
    id_cols = [c for c in df.columns if any(k in c.lower() for k in id_keywords)]
    
    # Identificar Target (logS)
    target_cols = [c for c in df.columns if c in ['logS']]

    cols_to_check = []
    
    # 1. Siempre chequear IDs (en cualquier modo)
    if id_cols:
        cols_to_check.extend(id_cols)
        
    # 2. En modo TRAIN, chequear también el Target (LogS)
    if args.mode == "train" and args.subset != "features_only":
        print("[Clean] Mode TRAIN: Checking IDs and Target (LogS). Allowing NaN features.")
        if target_cols:
            cols_to_check.extend(target_cols)
        else:
            print("[WARN] No target column found in TRAIN mode!")
            
    else:
        print("[Clean] Mode INFERENCE/TEST: Checking IDs only. Allowing NaN features/target.")

    # 3. Aplicar Filtro
    if cols_to_check:
        print(f"        Enforcing validity on: {cols_to_check}")
        df_clean = df.dropna(subset=cols_to_check, how='any')
    else:
        # Si no hay columnas críticas identificadas, no borramos nada (o borramos todo si somos paranoicos, mejor no borrar)
        df_clean = df

    n_final = len(df_clean)
    n_dropped = n_original - n_final
    
    print(f"[Clean] Rows: {n_original} -> {n_final} (Dropped {n_dropped})")

    if n_final == 0:
        print("[ERROR] All rows were dropped! Check data quality.")
        # No hacemos sys.exit para no romper pipelines en ejecución, pero generamos output vacío válido

    # Save
    df_clean.to_parquet(args.output, index=False)
    if args.save_csv:
        csv_out = os.path.splitext(args.output)[0] + ".csv"
        df_clean.to_csv(csv_out, index=False)

if __name__ == "__main__":
    main()