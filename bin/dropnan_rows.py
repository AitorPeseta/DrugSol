#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dropnan_rows.py
---------------
Final dataset cleaning step.
- Replaces empty strings/whitespace with NaN.
- TRAIN Mode: Drops rows with ANY NaNs (strict quality control).
- TEST Mode: Drops rows only if critical ID columns are missing.
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
    """Replaces whitespace-only strings with NaN."""
    # Select string columns
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(str_cols) > 0:
        # Vectorized replacement using regex
        # Replace empty strings or whitespace with NaN
        df[str_cols] = df[str_cols].replace(r'^\s*$', np.nan, regex=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Clean NaNs from dataset.")
    ap.add_argument("-i", "--input", required=True, help="Input file")
    ap.add_argument("-o", "--output", required=True, help="Output file")
    ap.add_argument("--mode", choices=["train", "test"], required=True, help="Cleaning mode")
    ap.add_argument("--save_csv", action="store_true", help="Save CSV copy")
    
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] File not found: {args.input}")

    # 1. Load
    print(f"[Clean] Loading {args.input}...")
    df = read_any(args.input)
    n_original = len(df)

    # 2. Standardize NaNs
    # Replace Infinite with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Replace empty strings with NaN
    df = coerce_empty_strings(df)

    # 3. Apply Cleaning Logic
    if args.mode == "train":
        print("[Clean] Mode TRAIN: Strict dropping of NaNs.")
        # For training, we generally want complete data.
        # Drop rows where ANY column is NaN (except maybe auxiliary info, but keep it simple)
        # Alternatively, you can specify subset=['logS', 'mordred__...']
        df_clean = df.dropna(how='any')
        
    else: # TEST mode
        print("[Clean] Mode TEST: Lenient dropping.")
        # For test/inference, we keep rows even if some features are missing (model might handle it or we impute later)
        # BUT we must drop rows if critical identifiers are missing
        critical_cols = [c for c in df.columns if 'smiles' in c.lower()]
        if critical_cols:
            df_clean = df.dropna(subset=critical_cols, how='any')
        else:
            df_clean = df

    n_final = len(df_clean)
    n_dropped = n_original - n_final
    
    print(f"[Clean] Rows: {n_original} -> {n_final} (Dropped {n_dropped})")

    if n_final == 0:
        sys.exit("[ERROR] All rows were dropped! Check your data quality.")

    # 4. Save
    df_clean.to_parquet(args.output, index=False)
    print(f"[Clean] Saved Parquet: {args.output}")

    if args.save_csv:
        csv_out = os.path.splitext(args.output)[0] + ".csv"
        df_clean.to_csv(csv_out, index=False)
        print(f"[Clean] Saved CSV: {csv_out}")

if __name__ == "__main__":
    main()