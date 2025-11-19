#!/usr/bin/env python3
# filter_by_temperature_range.py
# Filtra filas manteniendo solo las que tienen temperature_K dentro del rango especificado.

import argparse, os, sys
import pandas as pd
import numpy as np

def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)

def write_any(df: pd.DataFrame, path: str):
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    else:
        df.to_parquet(path)

def main():
    ap = argparse.ArgumentParser(description="Filtra filas por rango de temperatura (K).")
    ap.add_argument("--input", required=True, help="Archivo de entrada (.parquet o .csv)")
    ap.add_argument("--out", required=True, help="Archivo de salida (.parquet o .csv)")
    ap.add_argument("--temp-col", default="temp_C", help="Nombre de la columna de temperatura (default: temp_C)")
    ap.add_argument("--min-k", type=float, default=None, help="Temperatura mínima en K (inclusiva). Si no se indica, -inf.")
    ap.add_argument("--max-k", type=float, default=None, help="Temperatura máxima en K (inclusiva). Si no se indica, +inf.")
    ap.add_argument("--strict", action="store_true", help="Usar límites estrictos (min,max) en lugar de inclusivos [min,max].")
    ap.add_argument("--keep-na", action="store_true", help="Conservar filas con NaN en temperatura (por defecto se descartan).")
    args = ap.parse_args()

    df = read_any(args.input)
    if args.temp_col not in df.columns:
        sys.exit(f"[ERROR] La columna '{args.temp_col}' no existe en {args.input}.")

    # Columna como numérica
    t = pd.to_numeric(df[args.temp_col], errors="coerce")
    mask_valid_num = t.notna()

    # Límites
    lo = -np.inf if args.min_k is None else float(args.min_k)
    hi =  np.inf if args.max_k is None else float(args.max_k)

    if args.strict:
        in_range = (t > lo) & (t < hi)
    else:
        in_range = (t >= lo) & (t <= hi)

    if args.keep_na:
        # Mantén NaN + los que están en rango
        final_mask = in_range | (~mask_valid_num)
    else:
        # Solo los numéricos en rango
        final_mask = mask_valid_num & in_range

    kept = int(final_mask.sum())
    total = len(df)
    dropped = total - kept

    df_out = df.loc[final_mask].copy()
    # Escritura
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_any(df_out, args.out)

    # Resumen
    bounds = f"({'>' if args.strict else '>='} {lo} K) & ({'<' if args.strict else '<='} {hi} K)"
    print(f"[filter_by_temperature_range] OK  kept={kept}  dropped={dropped}  total={total}  rango: {bounds}")
    if not args.keep_na:
        n_na = int((~mask_valid_num).sum())
        if n_na:
            print(f"[AVISO] Descargadas {n_na} filas por temperatura NaN/no numérica. Usa --keep-na para conservarlas.", file=sys.stderr)

if __name__ == "__main__":
    main()
