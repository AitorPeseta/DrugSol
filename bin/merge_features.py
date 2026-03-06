#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Features: Combine Multiple Feature Sets
==============================================

Merges multiple feature DataFrames into a single unified dataset for model
training. Primary use case is combining traditional molecular descriptors
(Mordred) with learned embeddings (ChemBERTa).

Merge Strategy:
    Features are joined on a common ID column (default: row_uid). The merge
    type can be configured:
    - inner: Only keep rows present in ALL files (default, safest)
    - left:  Keep all rows from primary file, fill missing with NaN
    - outer: Keep all rows from all files, fill missing with NaN

Arguments:
    --primary          : Primary feature file (e.g., Mordred descriptors)
    --secondary        : Secondary feature file (e.g., ChemBERTa embeddings)
    --output           : Output Parquet file path
    --merge-col        : Column to join on (default: row_uid)
    --how              : Merge strategy: inner, left, outer (default: inner)
    --suffix-primary   : Suffix for duplicate columns from primary (default: '')
    --suffix-secondary : Suffix for duplicate columns from secondary (default: '_bert')
    --validate         : Validate row counts after merge

    Alternative mode (for 2+ files):
    --files            : Space-separated list of files to merge sequentially

Usage:
    # Merge two files
    python merge_features.py \\
        --primary train_mordred.parquet \\
        --secondary train_chemberta.parquet \\
        --output train_merged.parquet \\
        --merge-col row_uid \\
        --how inner \\
        --validate

    # Merge multiple files
    python merge_features.py \\
        --files file1.parquet file2.parquet file3.parquet \\
        --output merged.parquet \\
        --merge-col row_uid

Output:
    Parquet file containing all columns from input files, joined on merge_col.
    Duplicate column names (except merge_col) receive suffixes to avoid conflicts.

Notes:
    - The merge column must exist in all input files
    - Column order: merge_col first, then primary columns, then secondary columns
    - For inner join, output rows = intersection of input rows
    - Memory usage: approximately sum of input file sizes during merge
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_file(path: str) -> pd.DataFrame:
    """
    Read input file as DataFrame (supports Parquet and CSV).
    
    Args:
        path: File path (Parquet or CSV)
    
    Returns:
        Pandas DataFrame
    """
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep='\t')
    else:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)


def get_duplicate_columns(df1: pd.DataFrame, df2: pd.DataFrame, exclude: list) -> list:
    """
    Find column names that exist in both DataFrames.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        exclude: Columns to exclude from duplicate check
    
    Returns:
        List of duplicate column names
    """
    cols1 = set(df1.columns) - set(exclude)
    cols2 = set(df2.columns) - set(exclude)
    return list(cols1 & cols2)


def merge_two_dataframes(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    merge_col: str,
    how: str = 'inner',
    suffix_primary: str = '',
    suffix_secondary: str = '_bert'
) -> pd.DataFrame:
    """
    Merge two DataFrames on a common column.
    
    Args:
        df_primary: Primary DataFrame (left side of merge)
        df_secondary: Secondary DataFrame (right side of merge)
        merge_col: Column name to join on
        how: Merge type ('inner', 'left', 'outer')
        suffix_primary: Suffix for duplicate columns from primary
        suffix_secondary: Suffix for duplicate columns from secondary
    
    Returns:
        Merged DataFrame
    """
    # Validate merge column exists
    if merge_col not in df_primary.columns:
        raise ValueError(f"Merge column '{merge_col}' not found in primary file")
    if merge_col not in df_secondary.columns:
        raise ValueError(f"Merge column '{merge_col}' not found in secondary file")
    
    # SOLUCIÓN DEL ERROR: Encontrar y eliminar duplicados en el df_secondary
    duplicates = get_duplicate_columns(df_primary, df_secondary, [merge_col])
    
    if duplicates:
        print(f"[Merge] Found {len(duplicates)} duplicate columns (e.g., {duplicates[:5]})")
        print(f"[Merge] Dropping duplicates from secondary file to ensure a clean merge...")
        df_secondary = df_secondary.drop(columns=duplicates)
    
    # Perform clean merge without needing suffixes
    df_merged = pd.merge(
        df_primary,
        df_secondary,
        on=merge_col,
        how=how
    )
    
    return df_merged


def merge_multiple_dataframes(
    file_paths: list,
    merge_col: str,
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Sequentially merge multiple DataFrames.
    
    Args:
        file_paths: List of file paths to merge
        merge_col: Column name to join on
        how: Merge type
    
    Returns:
        Merged DataFrame
    """
    if len(file_paths) < 2:
        raise ValueError("Need at least 2 files to merge")
    
    print(f"[Merge] Loading {file_paths[0]}...")
    df_result = read_file(file_paths[0])
    
    for i, path in enumerate(file_paths[1:], start=2):
        print(f"[Merge] Merging file {i}/{len(file_paths)}: {path}...")
        df_next = read_file(path)
        
        df_result = merge_two_dataframes(
            df_result,
            df_next,
            merge_col=merge_col,
            how=how
        )
        print(f"[Merge] After merge {i}: {len(df_result):,} rows, {len(df_result.columns):,} columns")
    
    return df_result


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for feature merging script."""
    
    ap = argparse.ArgumentParser(
        description="Merge multiple feature sets into a single DataFrame."
    )
    
    # Two-file mode
    ap.add_argument("--primary", "-p",
                    help="Primary feature file (e.g., Mordred descriptors)")
    ap.add_argument("--secondary", "-s",
                    help="Secondary feature file (e.g., ChemBERTa embeddings)")
    
    # Multi-file mode
    ap.add_argument("--files", nargs='+',
                    help="Multiple files to merge sequentially")
    
    # Output
    ap.add_argument("--output", "-o", required=True,
                    help="Output Parquet file path")
    
    # Merge configuration
    ap.add_argument("--merge-col", default="row_uid",
                    help="Column to join on (default: row_uid)")
    ap.add_argument("--how", choices=["inner", "left", "outer"], default="inner",
                    help="Merge strategy (default: inner)")
    ap.add_argument("--suffix-primary", default="",
                    help="Suffix for duplicate columns from primary file")
    ap.add_argument("--suffix-secondary", default="_bert",
                    help="Suffix for duplicate columns from secondary file")
    
    # Validation
    ap.add_argument("--validate", action="store_true",
                    help="Validate that no rows are lost in merge")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Determine Mode and Validate Arguments
    # -------------------------------------------------------------------------
    if args.files:
        mode = "multi"
        if len(args.files) < 2:
            sys.exit("[ERROR] --files requires at least 2 files")
    elif args.primary and args.secondary:
        mode = "two"
    else:
        sys.exit("[ERROR] Must specify either --primary and --secondary, or --files")
    
    # -------------------------------------------------------------------------
    # Perform Merge
    # -------------------------------------------------------------------------
    if mode == "two":
        print(f"[Merge] Loading primary: {args.primary}...")
        df_primary = read_file(args.primary)
        print(f"[Merge] Loading secondary: {args.secondary}...")
        df_secondary = read_file(args.secondary)
        
        n_primary = len(df_primary)
        n_secondary = len(df_secondary)
        
        df_result = merge_two_dataframes(
            df_primary, df_secondary,
            merge_col=args.merge_col, how=args.how
        )
        
        if args.validate and args.how == "inner":
            expected = min(n_primary, n_secondary)
            if len(df_result) < expected * 0.95:
                print(f"[WARN] Significant row loss during merge!")
    else:
        df_result = merge_multiple_dataframes(args.files, merge_col=args.merge_col, how=args.how)
    
    # -------------------------------------------------------------------------
    # Reorder Columns (merge_col first)
    # -------------------------------------------------------------------------
    if args.merge_col in df_result.columns:
        cols = [args.merge_col] + [c for c in df_result.columns if c != args.merge_col]
        df_result = df_result[cols]
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    print(f"[Merge] Saving to {args.output}...")
    df_result.to_parquet(args.output, index=False)
    
    # -------------------------------------------------------------------------
    # Print Summary
    # -------------------------------------------------------------------------
    print(f"\n[Merge] Summary:")
    print(f"   -> Output rows: {len(df_result):,}")
    print(f"   -> Output columns: {len(df_result.columns):,}")

if __name__ == "__main__":
    main()
