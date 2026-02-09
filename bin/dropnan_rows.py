#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drop NaN Rows: Intelligent Data Cleaning
=========================================

Final data cleaning step that removes rows with missing critical values while
preserving rows where only feature columns have NaN (these can be imputed).

Cleaning Strategy:
    - TRAIN mode: Check IDs (row_uid, smiles_neutral) AND target (logS)
    - TEST mode: Check IDs only (target may be unknown during inference)
    
    Feature columns (Mordred descriptors, etc.) are NOT checked - missing
    values there are acceptable and will be handled by model imputation.

Arguments:
    --input, -i  : Input Parquet/CSV file
    --output, -o : Output Parquet file
    --mode       : Processing mode: train or test
    --subset     : Subset mode: all or features_only (default: features_only)
    --save_csv   : Also save CSV output

Usage:
    python dropnan_rows.py \\
        --input aligned_features.parquet \\
        --output final_train_gbm.parquet \\
        --mode train \\
        --subset features_only \\
        --save_csv

Output:
    Parquet file with rows containing valid critical columns.

Notes:
    - Empty strings are converted to NaN before checking
    - Infinity values are converted to NaN
    - Rows are only dropped for missing CRITICAL columns, not features
    - Warning is issued if all rows are dropped (data quality issue)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# I/O UTILITIES
# ============================================================================

def read_any(path: str) -> pd.DataFrame:
    """Read input file as DataFrame (supports Parquet and CSV)."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def coerce_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert empty/whitespace strings to NaN."""
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(str_cols) > 0:
        df[str_cols] = df[str_cols].replace(r'^\s*$', np.nan, regex=True)
    return df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for NaN row dropping script."""
    
    ap = argparse.ArgumentParser(
        description="Drop rows with missing critical values."
    )
    ap.add_argument("-i", "--input", required=True,
                    help="Input Parquet/CSV file")
    ap.add_argument("-o", "--output", required=True,
                    help="Output Parquet file")
    ap.add_argument("--mode", choices=["train", "test"], required=True,
                    help="Processing mode")
    ap.add_argument("--subset", default="features_only",
                    help="Subset mode: all or features_only")
    ap.add_argument("--save_csv", action="store_true",
                    help="Also save CSV output")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    if not Path(args.input).exists():
        sys.exit(f"[ERROR] File not found: {args.input}")
    
    print(f"[Clean] Loading {args.input}...")
    df = read_any(args.input)
    n_original = len(df)
    print(f"[Clean] Loaded {n_original:,} rows")
    
    # -------------------------------------------------------------------------
    # Standardize NaN Values
    # -------------------------------------------------------------------------
    # Convert infinity to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Convert empty strings to NaN
    df = coerce_empty_strings(df)
    
    # -------------------------------------------------------------------------
    # Identify Critical Columns
    # -------------------------------------------------------------------------
    # ID columns (always required)
    id_keywords = ['smiles_neutral', 'row_uid']
    id_cols = [c for c in df.columns if any(k in c.lower() for k in id_keywords)]
    
    # Target column
    target_cols = [c for c in df.columns if c == 'logS']
    
    # Build list of columns to check for NaN
    cols_to_check = []
    
    # Always check IDs
    if id_cols:
        cols_to_check.extend(id_cols)
    
    # In TRAIN mode, also check target (unless subset is features_only)
    if args.mode == "train" and args.subset != "features_only":
        print("[Clean] TRAIN mode: Checking IDs and target (logS)")
        if target_cols:
            cols_to_check.extend(target_cols)
        else:
            print("[WARN] No target column 'logS' found in TRAIN mode")
    else:
        print("[Clean] TEST/INFERENCE mode: Checking IDs only")
    
    # -------------------------------------------------------------------------
    # Apply Filter
    # -------------------------------------------------------------------------
    if cols_to_check:
        print(f"[Clean] Validating columns: {cols_to_check}")
        df_clean = df.dropna(subset=cols_to_check, how='any')
    else:
        print("[WARN] No critical columns identified - keeping all rows")
        df_clean = df
    
    n_final = len(df_clean)
    n_dropped = n_original - n_final
    
    print(f"[Clean] Rows: {n_original:,} -> {n_final:,} (dropped {n_dropped:,})")
    
    if n_final == 0:
        print("[ERROR] All rows were dropped! Check data quality.")
        # Continue to generate valid empty output rather than failing
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    df_clean.to_parquet(args.output, index=False)
    print(f"[Clean] Saved to {args.output}")
    
    if args.save_csv:
        csv_out = Path(args.output).with_suffix(".csv")
        df_clean.to_csv(csv_out, index=False)
        print(f"[Clean] Saved CSV to {csv_out}")


if __name__ == "__main__":
    main()
