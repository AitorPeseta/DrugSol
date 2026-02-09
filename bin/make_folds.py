#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Folds: Cross-Validation Fold Generation with Grouped Stratification
=========================================================================

Generates cross-validation fold assignments using grouped stratified K-fold
splitting. This ensures both scaffold separation (prevents data leakage) and
balanced target distribution across folds.

Algorithm Overview:
    1. Group molecules by scaffold cluster (from Butina clustering)
    2. Create stratification labels from target (logS) bins
    3. Collapse to group level using majority label per cluster
    4. Clean rare classes that would cause StratifiedKFold to fail
    5. Apply StratifiedKFold on group level (with fallback to KFold)
    6. Map fold assignments back to individual molecules
    7. Handle orphan rows with random assignment (safety net)

Arguments:
    --input        : Input Parquet/CSV file with training data
    --out          : Output Parquet file path
    --id-col       : Unique identifier column (default: row_uid)
    --group-col    : Scaffold cluster column (default: cluster_ecfp4_0p7)
    --target       : Target column for stratification (default: logS)
    --n-splits     : Number of CV folds (default: 5)
    --bins         : Number of bins for target stratification (default: 5)
    --seed         : Random seed for reproducibility (default: 42)
    --strat-mode   : Stratification mode (legacy, accepted but ignored)
    --temp-col     : Temperature column (legacy, accepted but ignored)
    --temp-step    : Temperature binning step (legacy, accepted but ignored)
    --temp-unit    : Temperature unit (legacy, accepted but ignored)

Usage:
    python make_folds.py \\
        --input train_data.parquet \\
        --out folds.parquet \\
        --id-col row_uid \\
        --group-col cluster_ecfp4_0p7 \\
        --target logS \\
        --n-splits 5 \\
        --bins 5 \\
        --seed 42

Output:
    Parquet file with columns:
    - {id_col}: Original row identifier
    - fold: Assigned fold number (0 to n_splits-1)
    - {group_col}: Scaffold cluster ID (if present in input)

Notes:
    - Stratification by both temperature and logS was found to be too restrictive
      for 5-fold CV, causing dropped rows. Current implementation uses only logS.
    - Rare classes (count < n_splits) are merged into an "OTHER" category
    - Any rows that fail assignment are randomly distributed (safety fallback)
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from collections import Counter


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_any(path: str) -> pd.DataFrame:
    """
    Read input file as DataFrame (supports Parquet and CSV).
    
    Args:
        path: File path (Parquet or CSV)
    
    Returns:
        Pandas DataFrame
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def get_strat_label(df: pd.DataFrame, target_col: str, n_bins: int = 5) -> np.ndarray:
    """
    Create stratification labels by binning the target column.
    
    Uses quantile-based binning (qcut) for even distribution across bins.
    Falls back to equal-width binning (cut) if qcut fails due to duplicates.
    
    Args:
        df: Input DataFrame
        target_col: Target column name for stratification
        n_bins: Number of bins to create
    
    Returns:
        NumPy array of bin labels (0 to n_bins-1, or -1 for missing)
    """
    if target_col not in df.columns:
        print(f"[WARN] Target column '{target_col}' not found. Using uniform labels.")
        return np.zeros(len(df), dtype=int)
    
    y = pd.to_numeric(df[target_col], errors='coerce')
    
    # Try quantile-based binning first (equal sample counts per bin)
    try:
        labels = pd.qcut(y, n_bins, labels=False, duplicates='drop')
        return labels.fillna(-1).astype(int)
    except ValueError:
        pass
    
    # Fallback to equal-width binning (equal value ranges per bin)
    try:
        labels = pd.cut(y, n_bins, labels=False, duplicates='drop')
        return labels.fillna(-1).astype(int)
    except ValueError:
        print(f"[WARN] Could not create {n_bins} bins for '{target_col}'. Using uniform labels.")
        return np.zeros(len(df), dtype=int)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for fold generation script."""
    
    ap = argparse.ArgumentParser(
        description="Generate cross-validation folds with grouped stratification."
    )
    
    # Required arguments
    ap.add_argument("--input", required=True,
                    help="Input Parquet/CSV file")
    ap.add_argument("--out", required=True,
                    help="Output Parquet file")
    
    # Column configuration
    ap.add_argument("--id-col", default="row_uid",
                    help="Unique identifier column (default: row_uid)")
    ap.add_argument("--group-col", default="cluster_ecfp4_0p7",
                    help="Scaffold cluster column (default: cluster_ecfp4_0p7)")
    ap.add_argument("--target", default="logS",
                    help="Target column for stratification (default: logS)")
    
    # CV configuration
    ap.add_argument("--n-splits", type=int, default=5,
                    help="Number of CV folds (default: 5)")
    ap.add_argument("--bins", type=int, default=5,
                    help="Number of bins for stratification (default: 5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    
    # Legacy arguments (accepted for compatibility but ignored)
    ap.add_argument("--strat-mode", default="both",
                    help="[LEGACY] Stratification mode (ignored)")
    ap.add_argument("--temp-col", default="temp_C",
                    help="[LEGACY] Temperature column (ignored)")
    ap.add_argument("--temp-step", type=int, default=2,
                    help="[LEGACY] Temperature binning step (ignored)")
    ap.add_argument("--temp-unit", default="auto",
                    help="[LEGACY] Temperature unit (ignored)")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Input Data
    # -------------------------------------------------------------------------
    print(f"[Folds] Loading {args.input}...")
    df = read_any(args.input)
    print(f"[Folds] Loaded {len(df):,} rows")
    
    # -------------------------------------------------------------------------
    # Step 1: Prepare Grouping Column
    # -------------------------------------------------------------------------
    if args.group_col not in df.columns:
        print(f"[WARN] Group column '{args.group_col}' not found. Each row is its own group.")
        df["_grp"] = df.index
    else:
        # Fill missing cluster IDs with "UNKNOWN"
        df["_grp"] = df[args.group_col].fillna("UNKNOWN")
    
    # -------------------------------------------------------------------------
    # Step 2: Create Stratification Labels
    # -------------------------------------------------------------------------
    # Note: We stratify only by target (logS), not by temperature.
    # Stratifying by both was too restrictive for 5-fold CV and caused dropped rows.
    df["_strata"] = get_strat_label(df, args.target, n_bins=args.bins)
    
    # -------------------------------------------------------------------------
    # Step 3: Collapse to Group Level
    # -------------------------------------------------------------------------
    # Each scaffold cluster gets assigned the majority stratification label
    # This ensures all molecules in a cluster go to the same fold
    grp_meta = df.groupby("_grp")["_strata"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    ).reset_index()
    
    groups = grp_meta["_grp"].values
    labels = grp_meta["_strata"].values
    
    n_groups = len(groups)
    print(f"[Folds] Found {n_groups:,} unique scaffold groups")
    
    # -------------------------------------------------------------------------
    # Step 4: Clean Rare Classes
    # -------------------------------------------------------------------------
    # StratifiedKFold requires at least n_splits samples per class
    # Rare classes are merged into a catch-all category (-1)
    counts = Counter(labels)
    rare_classes = [cls for cls, count in counts.items() if count < args.n_splits]
    
    if rare_classes:
        n_affected = sum(counts[cls] for cls in rare_classes)
        print(f"[Folds] Merging {len(rare_classes)} rare classes ({n_affected:,} groups) into 'OTHER'")
    
    labels_clean = np.array([
        lbl if counts[lbl] >= args.n_splits else -1
        for lbl in labels
    ])
    
    # -------------------------------------------------------------------------
    # Step 5: Apply K-Fold Splitting
    # -------------------------------------------------------------------------
    folds_map = {}
    
    try:
        # Primary: Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed
        )
        split_gen = skf.split(groups, labels_clean)
        split_mode = "Stratified"
    except ValueError as e:
        # Fallback: Simple K-Fold (still respects groups)
        print(f"[WARN] StratifiedKFold failed: {e}")
        print("[Folds] Falling back to simple KFold")
        kf = KFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed
        )
        split_gen = kf.split(groups)
        split_mode = "Random (Fallback)"
    
    # Assign folds to groups
    for fold_idx, (_, test_idx) in enumerate(split_gen):
        for idx in test_idx:
            grp_id = groups[idx]
            folds_map[grp_id] = fold_idx
    
    # -------------------------------------------------------------------------
    # Step 6: Map Folds Back to Rows
    # -------------------------------------------------------------------------
    df["fold"] = df["_grp"].map(folds_map)
    
    # -------------------------------------------------------------------------
    # Step 7: Safety Net for Orphan Rows
    # -------------------------------------------------------------------------
    # In rare edge cases, some rows might not get assigned a fold
    # (e.g., due to floating-point issues or unexpected data)
    if df["fold"].isna().any():
        n_missing = df["fold"].isna().sum()
        print(f"[WARN] {n_missing:,} rows without fold assignment. Assigning randomly.")
        
        rng = np.random.default_rng(args.seed)
        df.loc[df["fold"].isna(), "fold"] = rng.integers(
            0, args.n_splits, size=n_missing
        )
    
    df["fold"] = df["fold"].astype(int)
    
    # -------------------------------------------------------------------------
    # Print Summary
    # -------------------------------------------------------------------------
    fold_dist = df["fold"].value_counts(normalize=True).sort_index()
    print(f"\n[Folds] Generated {args.n_splits} folds ({split_mode})")
    print("[Folds] Fold distribution:")
    for fold, pct in fold_dist.items():
        count = (df["fold"] == fold).sum()
        print(f"   Fold {fold}: {count:>6,} samples ({pct*100:>5.1f}%)")
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    # Save only essential columns
    out_cols = [args.id_col, "fold"]
    if args.group_col in df.columns:
        out_cols.append(args.group_col)
    
    # Validate id column exists
    if args.id_col not in df.columns:
        print(f"[WARN] ID column '{args.id_col}' not found. Using index.")
        df[args.id_col] = df.index
    
    df[out_cols].to_parquet(args.out, index=False)
    print(f"\n[Folds] Saved to {args.out}")


if __name__ == "__main__":
    main()
