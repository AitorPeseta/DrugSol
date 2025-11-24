#!/usr/bin/env python3
# filter_outlier.py
# Removes rows marked as outliers (is_outlier == 1).
# Optional: Drops the 'is_outlier' column after filtering to clean up the schema.

import pandas as pd
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Filter rows that are NOT outliers (is_outlier == 0) and save result."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input file (.parquet or .csv) containing 'is_outlier' column.")
    parser.add_argument("--out", required=True,
                        help="Output file path (.parquet).")
    parser.add_argument("--save-csv", action="store_true",
                        help="If set, also saves a CSV version.")
    
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        sys.exit(f"[ERROR] File not found: {input_path}")

    # 1. Load Data
    print(f"[Filter Outlier] Loading {input_path}...")
    try:
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read file: {e}")

    if "is_outlier" not in df.columns:
        sys.exit("[ERROR] Column 'is_outlier' not found in dataset.")

    # 2. Statistics
    total_rows = len(df)
    n_outliers = df["is_outlier"].sum()
    
    # 3. Filter (Keep only is_outlier == 0)
    # We create a copy to avoid SettingWithCopy warnings
    df_filtered = df[df["is_outlier"] == 0].copy()
    
    df_filtered.drop(columns=["is_outlier"], inplace=True)
    
    final_rows = len(df_filtered)

    # 4. Save Parquet
    df_filtered.to_parquet(args.out, index=False)
    print(f"[Filter Outlier] Saved Parquet: {args.out}")

    # 5. Save CSV (Optional)
    if args.save_csv:
        csv_out = os.path.splitext(args.out)[0] + ".csv"
        df_filtered.to_csv(csv_out, index=False, encoding="utf-8", lineterminator="\n")
        print(f"[Filter Outlier] Saved CSV: {csv_out}")

    # Summary
    print(f"[Filter Outlier] Summary: Input={total_rows} | Removed={int(n_outliers)} | Kept={final_rows}")

if __name__ == "__main__":
    main()