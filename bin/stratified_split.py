#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified Split: Scaffold-Aware Train/Test Partitioning
=========================================================

Generates a single train/test split ensuring scaffold separation and
stratified target distribution. Molecules within the same scaffold cluster
are kept together to prevent data leakage from structurally similar compounds.

Splitting Strategy:
    1. Build stratification labels from temperature and solubility bins
    2. Group molecules by scaffold cluster
    3. Assign each group a label based on majority vote
    4. Merge rare stratification classes to ensure splittability
    5. Distribute groups to train/test while preserving stratification
    6. Map assignments back to individual molecules

Arguments:
    --input               : Input Parquet file with balanced data
    --group-col           : Scaffold cluster column (default: cluster_ecfp4_0p7)
    --temp-col            : Temperature column (default: temp_C)
    --target-col          : Target column for stratification (default: logS)
    --temp-step           : Temperature binning step in °C (default: 5.0)
    --test-size           : Fraction of data for test set (default: 0.2)
    --seed                : Random seed for reproducibility (default: 42)
    --min-groups-per-class: Minimum groups required per stratum (default: 2)
    --outdir              : Output directory (default: current directory)

Usage:
    python stratified_split.py \\
        --input balanced_data.parquet \\
        --group-col cluster_ecfp4_0p7 \\
        --temp-col temp_C \\
        --temp-step 5 \\
        --test-size 0.2 \\
        --seed 42 \\
        --outdir .

Output:
    - train.parquet: Training set (~80% of molecules)
    - test.parquet: Test set (~20% of molecules)
    
    The actual ratio may vary slightly due to scaffold group constraints.

Notes:
    - Scaffold splitting prevents leakage from structurally similar molecules
    - Stratification ensures balanced logS and temperature distributions
    - Rare strata (< min-groups-per-class) are merged into "OTHER" category
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================================
# STRATIFICATION UTILITIES
# ============================================================================

def build_strat_label(
    df: pd.DataFrame,
    temp_col: str = "temp_C",
    temp_step: float = 5.0,
    target_col: str = "logS",
    n_bins: int = 5
) -> pd.Series:
    """
    Build stratification labels from temperature and target columns.
    
    Creates composite labels "temp_bin|target_bin" for stratified splitting.
    
    Args:
        df: Input DataFrame
        temp_col: Temperature column name
        temp_step: Temperature binning step in degrees
        target_col: Target column name (solubility)
        n_bins: Number of quantile bins for target
    
    Returns:
        Series of stratification labels
    """
    # Temperature binning (round to nearest step)
    if temp_col in df.columns:
        t = pd.to_numeric(df[temp_col], errors="coerce")
        t_bin = (np.round(t / temp_step) * temp_step).fillna(-999).astype(int).astype(str)
    else:
        t_bin = pd.Series("NoTemp", index=df.index)
    
    # Target binning (quantiles)
    if target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce")
        try:
            y_bin = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
            y_bin = y_bin.fillna(-1).astype(int).astype(str)
        except ValueError:
            y_bin = pd.Series("All", index=df.index)
    else:
        y_bin = pd.Series("NoTarget", index=df.index)
    
    return t_bin + "|" + y_bin


def pick_group_label(labels: list) -> str:
    """
    Select the majority label for a group.
    
    Args:
        labels: List of stratification labels
    
    Returns:
        Most common label in the list
    """
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def collapse_rare_classes(
    group_df: pd.DataFrame,
    label_col: str = "group_strat_label",
    min_groups: int = 2
) -> pd.Series:
    """
    Merge rare stratification classes into "OTHER" category.
    
    Classes with fewer than min_groups members cannot be reliably split,
    so they are merged into a catch-all category.
    
    Args:
        group_df: DataFrame with group-level stratification labels
        label_col: Column containing labels
        min_groups: Minimum group count to keep a class
    
    Returns:
        Series with rare classes replaced by "OTHER"
    """
    counts = group_df[label_col].value_counts()
    rare_labels = counts[counts < min_groups].index
    return group_df[label_col].mask(group_df[label_col].isin(rare_labels), "OTHER")


# ============================================================================
# SPLITTING ALGORITHM
# ============================================================================

def smart_balanced_split(
    grp_meta: pd.DataFrame,
    test_size: float,
    seed: int
) -> tuple:
    """
    Split scaffold groups into train/test while preserving stratification.
    
    Uses a greedy algorithm that iterates through strata and assigns groups
    to train or test to best match the target test_size ratio.
    
    Args:
        grp_meta: DataFrame with group metadata (size, strat label)
        test_size: Target fraction for test set
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (train_group_indices, test_group_indices)
    """
    rng = np.random.default_rng(seed)
    strata = grp_meta["group_strat_label"].unique()
    
    train_groups_all = []
    test_groups_all = []
    
    for stratum in strata:
        subset = grp_meta[grp_meta["group_strat_label"] == stratum]
        indices = subset.index.values
        sizes = subset["size"].values
        
        total_rows = sizes.sum()
        target_test_rows = int(total_rows * test_size)
        
        # Shuffle groups within stratum
        perm = rng.permutation(len(indices))
        indices = indices[perm]
        sizes = sizes[perm]
        
        # Greedy assignment to minimize deviation from target
        current_test_rows = 0
        test_g_idxs = []
        train_g_idxs = []
        
        for g_idx, sz in zip(indices, sizes):
            dist_if_add = abs((current_test_rows + sz) - target_test_rows)
            dist_if_skip = abs(current_test_rows - target_test_rows)
            
            if dist_if_add < dist_if_skip:
                test_g_idxs.append(g_idx)
                current_test_rows += sz
            else:
                # Ensure at least one group in test if stratum is non-empty
                if current_test_rows == 0 and len(test_g_idxs) == 0:
                    test_g_idxs.append(g_idx)
                    current_test_rows += sz
                else:
                    train_g_idxs.append(g_idx)
        
        train_groups_all.extend(train_g_idxs)
        test_groups_all.extend(test_g_idxs)
    
    return train_groups_all, test_groups_all


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for stratified splitting script."""
    
    ap = argparse.ArgumentParser(
        description="Scaffold-aware stratified train/test splitting."
    )
    ap.add_argument("--input", required=True,
                    help="Input Parquet file")
    ap.add_argument("--group-col", default="cluster_ecfp4_0p7",
                    help="Scaffold cluster column (default: cluster_ecfp4_0p7)")
    ap.add_argument("--temp-col", default="temp_C",
                    help="Temperature column (default: temp_C)")
    ap.add_argument("--target-col", default="logS",
                    help="Target column (default: logS)")
    ap.add_argument("--temp-step", type=float, default=5.0,
                    help="Temperature binning step (default: 5.0)")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="Test set fraction (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--min-groups-per-class", type=int, default=2,
                    help="Minimum groups per stratum (default: 2)")
    ap.add_argument("--outdir", default=".",
                    help="Output directory (default: current)")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"[Split] Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"[Split] Loaded {len(df):,} molecules")
    
    # -------------------------------------------------------------------------
    # Build Stratification Labels
    # -------------------------------------------------------------------------
    print("[Split] Building stratification labels...")
    df["_strat_label"] = build_strat_label(
        df,
        temp_col=args.temp_col,
        temp_step=args.temp_step,
        target_col=args.target_col
    )
    
    # -------------------------------------------------------------------------
    # Aggregate to Group Level
    # -------------------------------------------------------------------------
    print(f"[Split] Grouping by scaffold column: {args.group_col}")
    
    grp_stats = df.groupby(args.group_col)["_strat_label"].agg([
        ('size', 'count'),
        ('group_strat_label', lambda x: pick_group_label(list(x)))
    ]).reset_index()
    
    n_groups = len(grp_stats)
    print(f"[Split] Found {n_groups:,} scaffold groups")
    
    # Collapse rare classes
    grp_stats["group_strat_label"] = collapse_rare_classes(
        grp_stats,
        "group_strat_label",
        args.min_groups_per_class
    )
    
    n_strata = grp_stats["group_strat_label"].nunique()
    print(f"[Split] Using {n_strata} stratification classes")
    
    # -------------------------------------------------------------------------
    # Perform Split
    # -------------------------------------------------------------------------
    print(f"[Split] Generating {1-args.test_size:.0%}/{args.test_size:.0%} split (seed={args.seed})...")
    
    train_idx_g, test_idx_g = smart_balanced_split(
        grp_stats,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Map back to molecule level
    train_grps = set(grp_stats.iloc[train_idx_g][args.group_col])
    test_grps = set(grp_stats.iloc[test_idx_g][args.group_col])
    
    mask_train = df[args.group_col].isin(train_grps)
    mask_test = df[args.group_col].isin(test_grps)
    
    df_train = df[mask_train].drop(columns=["_strat_label"])
    df_test = df[mask_test].drop(columns=["_strat_label"])
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    train_path = out_path / "train.parquet"
    test_path = out_path / "test.parquet"
    
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    
    # -------------------------------------------------------------------------
    # Report Results
    # -------------------------------------------------------------------------
    n_train = len(df_train)
    n_test = len(df_test)
    ratio = n_test / (n_train + n_test) if (n_train + n_test) > 0 else 0
    
    print(f"\n[Split] Results:")
    print(f"        Train: {n_train:,} molecules ({len(train_grps):,} scaffolds)")
    print(f"        Test:  {n_test:,} molecules ({len(test_grps):,} scaffolds)")
    print(f"        Test ratio: {ratio:.1%} (target: {args.test_size:.0%})")
    print(f"\n[Split] Saved to {out_path}/")


if __name__ == "__main__":
    main()
