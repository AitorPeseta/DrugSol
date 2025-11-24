#!/usr/bin/env python3
import argparse
import sys
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Filter dataset to keep only aqueous solubility entries.")
    ap.add_argument("--input", required=True, help="Path to input parquet file")
    ap.add_argument("--output", required=True, help="Path to output parquet file")
    return ap.parse_args()

def main():
    args = parse_args()

    print(f"[Filter Water] Loading {args.input}...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to read input file: {e}")
        sys.exit(1)

    initial_rows = len(df)

    # Filter logic: Case-insensitive check for 'water'
    # We strip whitespace just in case ' water ' slipped through
    df_water = df[df["solvent"].astype(str).str.strip().str.lower() == "water"]
    
    final_rows = len(df_water)
    dropped = initial_rows - final_rows

    print(f"[Filter Water] Rows before: {initial_rows}")
    print(f"[Filter Water] Rows after : {final_rows}")
    print(f"[Filter Water] Dropped (non-water): {dropped}")

    if final_rows == 0:
        print("[WARNING] Resulting dataset is empty! Check if 'solvent' column contains 'water'.")

    # Save output
    df_water.to_parquet(args.output, index=False)
    print(f"[Filter Water] Saved to {args.output}")

if __name__ == "__main__":
    main()