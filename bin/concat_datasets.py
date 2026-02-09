#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concat Datasets: Combine Train and Test Data
=============================================

Concatenates train and test datasets into a single full dataset for final
model training after validation is complete.

Operations:
    1. Load train and test parquet files
    2. Concatenate vertically (append rows)
    3. Remove 'fold' column (no longer meaningful)
    4. Remove duplicates by row_uid (safety measure)
    5. Save combined dataset

Use Case:
    After cross-validation confirms model quality, we retrain on the full
    dataset (train + test) to maximize data available for production model.

Arguments:
    --train : Training data parquet file
    --test  : Test data parquet file
    --out   : Output parquet file path

Usage:
    python concat_datasets.py \\
        --train train_data.parquet \\
        --test test_data.parquet \\
        --out full_dataset.parquet

Output:
    Combined parquet file with all samples
"""

import argparse
import sys

import pandas as pd


def main():
    """Main entry point for dataset concatenation."""
    
    ap = argparse.ArgumentParser(
        description="Concatenate train and test datasets."
    )
    ap.add_argument("--train", required=True,
                    help="Training data parquet file")
    ap.add_argument("--test", required=True,
                    help="Test data parquet file")
    ap.add_argument("--out", required=True,
                    help="Output parquet file")
    
    args = ap.parse_args()
    
    print(f"[Concat] Merging {args.train} and {args.test}...")
    
    try:
        # Load datasets
        df_train = pd.read_parquet(args.train)
        df_test = pd.read_parquet(args.test)
        
        print(f"[Concat] Train samples: {len(df_train):,}")
        print(f"[Concat] Test samples: {len(df_test):,}")
        
        # Concatenate
        df_full = pd.concat([df_train, df_test], ignore_index=True)
        
        # Remove 'fold' column if present (no longer meaningful)
        if "fold" in df_full.columns:
            df_full = df_full.drop(columns=["fold"])
            print("[Concat] Removed 'fold' column")
        
        # Remove duplicates by row_uid (safety measure)
        if "row_uid" in df_full.columns:
            n_before = len(df_full)
            df_full = df_full.drop_duplicates(subset=["row_uid"])
            n_removed = n_before - len(df_full)
            if n_removed > 0:
                print(f"[Concat] Removed {n_removed} duplicate rows")
        
        # Save
        df_full.to_parquet(args.out, index=False)
        
        print(f"[Concat] Total samples: {len(df_full):,}")
        print(f"[Concat] Saved to: {args.out}")
        
    except Exception as e:
        print(f"[ERROR] Failed to concatenate: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
