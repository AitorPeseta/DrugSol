#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align Feature Columns: Test-to-Train Column Alignment
======================================================

Aligns the columns of a test dataset to exactly match the training dataset.
Critical for ensuring consistent feature space between training and inference.

Operations Performed:
    1. Drop columns present in test but not in train (unseen features)
    2. Add columns present in train but not in test (filled appropriately)
    3. Reorder columns to match training exactly
    4. Fill missing one-hot encoded columns with 0 (not NaN)
    5. Fill missing numeric columns with NaN (handled by model's imputer)

Arguments:
    --train        : Reference training file (defines expected columns)
    --test         : Test file to align
    --out          : Output aligned test file
    --onehot-prefix: Prefix for one-hot columns to fill with 0 (default: solv_)

Usage:
    python align_feature_columns.py \\
        --train train_features_filtered.parquet \\
        --test test_features_filtered.parquet \\
        --out features_test_aligned.parquet \\
        --onehot-prefix "solv_"

Output:
    Parquet file with columns matching training file exactly:
    - Same columns (no extra, no missing)
    - Same order
    - Appropriate fill values for added columns

Notes:
    - One-hot encoded columns (matching prefix) are filled with 0
    - Other numeric columns are filled with NaN
    - Original test data is preserved for existing columns
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


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for column alignment script."""
    
    ap = argparse.ArgumentParser(
        description="Align test dataset columns to match training dataset."
    )
    ap.add_argument("--train", required=True,
                    help="Reference training file")
    ap.add_argument("--test", required=True,
                    help="Test file to align")
    ap.add_argument("--out", required=True,
                    help="Output aligned test file")
    ap.add_argument("--onehot-prefix", default="solv_",
                    help="Prefix for one-hot columns (filled with 0)")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"[Align] Loading reference (train): {args.train}")
    df_train = read_any(args.train)
    
    print(f"[Align] Loading target (test): {args.test}")
    df_test = read_any(args.test)
    
    cols_train = df_train.columns
    cols_test_orig = df_test.columns
    
    print(f"[Align] Train columns: {len(cols_train)}")
    print(f"[Align] Test columns: {len(cols_test_orig)}")
    
    # -------------------------------------------------------------------------
    # Vectorized Alignment
    # -------------------------------------------------------------------------
    # reindex performs: drop extras, add missing (as NaN), reorder
    # All in one optimized operation
    print("[Align] Reindexing test to match train...")
    df_aligned = df_test.reindex(columns=cols_train)
    
    # -------------------------------------------------------------------------
    # Smart Fill for Added Columns
    # -------------------------------------------------------------------------
    # One-hot encoded columns should be 0 (absence), not NaN
    onehot_cols = [c for c in df_aligned.columns if c.startswith(args.onehot_prefix)]
    
    if onehot_cols:
        # Count how many were actually missing (now NaN)
        missing_onehot = [c for c in onehot_cols if c not in cols_test_orig]
        if missing_onehot:
            print(f"[Align] Filling {len(missing_onehot)} missing one-hot columns with 0")
        
        # Fill NaN with 0 for all one-hot columns (preserves existing 0s and 1s)
        df_aligned[onehot_cols] = df_aligned[onehot_cols].fillna(0).astype("int8")
    
    # -------------------------------------------------------------------------
    # Report Changes
    # -------------------------------------------------------------------------
    added = set(cols_train) - set(cols_test_orig)
    removed = set(cols_test_orig) - set(cols_train)
    
    if added:
        print(f"[Align] Added {len(added)} missing columns")
        if len(added) <= 5:
            print(f"        Columns: {list(added)}")
        else:
            print(f"        Examples: {list(added)[:3]}...")
    
    if removed:
        print(f"[Align] Removed {len(removed)} extra columns")
        if len(removed) <= 5:
            print(f"        Columns: {list(removed)}")
        else:
            print(f"        Examples: {list(removed)[:3]}...")
    
    if not added and not removed:
        print("[Align] Columns already aligned (reordered only)")
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_aligned.to_parquet(args.out, index=False)
    
    print(f"\n[Align] Saved aligned test to: {args.out}")
    print(f"        Final shape: {df_aligned.shape}")


if __name__ == "__main__":
    main()
