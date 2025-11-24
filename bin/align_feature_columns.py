#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_feature_columns.py
------------------------
Aligns the columns of the Test dataset to match the Train dataset exactly.
1. Drops columns present in Test but not in Train.
2. Adds columns present in Train but not in Test (filled with NaN).
3. Fills missing One-Hot encoded columns (e.g. 'solv_') with 0.
4. Enforces exact column order.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser(description="Align Test columns to Train.")
    ap.add_argument("--train", required=True, help="Reference Train file")
    ap.add_argument("--test", required=True, help="Target Test file to align")
    ap.add_argument("--out", required=True, help="Output aligned Test file")
    args = ap.parse_args()

    print(f"[Align] Loading Train: {args.train}")
    df_train = read_any(args.train)
    
    print(f"[Align] Loading Test: {args.test}")
    df_test = read_any(args.test)

    cols_train = df_train.columns
    cols_test_orig = df_test.columns

    print(f"[Align] Train cols: {len(cols_train)} | Test cols: {len(cols_test_orig)}")

    # 1. VECTORIZED ALIGNMENT (The Magic Step)
    # reindex does: Drop extras, Add missing (as NaN), Reorder. All in one C-optimized call.
    print("[Align] Reindexing Test to match Train...")
    df_aligned = df_test.reindex(columns=cols_train)

    # 2. Smart Fill
    # Identify columns that were added (missing in original test)
    # and columns that are One-Hot (start with 'solv_')
    
    # For solvent columns that were missing, fill with 0 instead of NaN
    # (If the solvent wasn't in the test set, its boolean flag is 0)
    solv_cols = [c for c in df_aligned.columns if c.startswith("solv_")]
    
    if solv_cols:
        print(f"[Align] Filling NaNs in {len(solv_cols)} solvent columns with 0...")
        # We only fill NaNs, preserving existing 1s/0s
        df_aligned[solv_cols] = df_aligned[solv_cols].fillna(0).astype("int8")

    # 3. Report Changes
    added = set(cols_train) - set(cols_test_orig)
    removed = set(cols_test_orig) - set(cols_train)
    
    if added:
        print(f"[Align] Added {len(added)} missing columns (filled NaN/0). Example: {list(added)[:3]}")
    if removed:
        print(f"[Align] Removed {len(removed)} extra columns. Example: {list(removed)[:3]}")

    # 4. Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_aligned.to_parquet(args.out, index=False)
    print(f"[Align] Saved aligned Test to: {args.out}")

if __name__ == "__main__":
    main()