#!/usr/bin/env python3
# filter_by_temperature_range.py
# Filters rows keeping only those where the temperature column is within a specified range.

import argparse
import os
import sys
import pandas as pd
import numpy as np

def read_any(path: str) -> pd.DataFrame:
    """Reads CSV or Parquet files."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)

def write_any(df: pd.DataFrame, path: str):
    """Writes to CSV or Parquet based on extension."""
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    else:
        df.to_parquet(path)

def main():
    ap = argparse.ArgumentParser(description="Filter rows by a numeric range in a specific column.")
    ap.add_argument("--input", required=True, help="Input file (.parquet or .csv)")
    ap.add_argument("--out", required=True, help="Output file (.parquet or .csv)")
    ap.add_argument("--temp-col", default="temp_C", help="Name of the temperature column (default: temp_C)")
    
    # Changed from min-k to min to be unit-agnostic (since we work with Celsius)
    ap.add_argument("--min", type=float, default=None, help="Minimum value (inclusive). Default: -inf")
    ap.add_argument("--max", type=float, default=None, help="Maximum value (inclusive). Default: +inf")
    
    ap.add_argument("--strict", action="store_true", help="Use strict inequality (min < x < max) instead of inclusive.")
    ap.add_argument("--keep-na", action="store_true", help="Keep rows with NaN in the target column.")
    
    args = ap.parse_args()

    # 1. Load Data
    print(f"[Filter Temp] Loading {args.input}...")
    try:
        df = read_any(args.input)
    except Exception as e:
        sys.exit(f"[ERROR] Could not read file: {e}")

    if args.temp_col not in df.columns:
        sys.exit(f"[ERROR] Column '{args.temp_col}' does not exist in the dataset.")

    # 2. Convert column to numeric (force coercion)
    t = pd.to_numeric(df[args.temp_col], errors="coerce")
    mask_valid_num = t.notna()

    # 3. Define Limits
    lo = -np.inf if args.min is None else float(args.min)
    hi =  np.inf if args.max is None else float(args.max)

    # 4. Create Boolean Mask
    if args.strict:
        in_range = (t > lo) & (t < hi)
    else:
        in_range = (t >= lo) & (t <= hi)

    if args.keep_na:
        # Keep if in range OR if it was NaN originally
        final_mask = in_range | (~mask_valid_num)
    else:
        # Keep only if valid number AND in range
        final_mask = mask_valid_num & in_range

    # 5. Stats
    kept = int(final_mask.sum())
    total = len(df)
    dropped = total - kept
    
    # 6. Filter and Save
    df_out = df.loc[final_mask].copy()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_any(df_out, args.out)

    # Summary Log
    operator_min = '>' if args.strict else '>='
    operator_max = '<' if args.strict else '<='
    bounds_desc = f"({operator_min} {lo}) & ({operator_max} {hi})"
    
    print(f"[Filter Temp] OK. Kept: {kept} | Dropped: {dropped} | Total: {total}")
    print(f"[Filter Temp] Logic: Column '{args.temp_col}' {bounds_desc}")

    if not args.keep_na:
        n_na = int((~mask_valid_num).sum())
        if n_na > 0:
            print(f"[WARNING] Dropped {n_na} rows due to NaN/Non-numeric temperature.", file=sys.stderr)

if __name__ == "__main__":
    main()