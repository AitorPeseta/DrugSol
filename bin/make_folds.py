#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_folds.py
-------------
Generates Out-of-Fold (OOF) indices for Cross-Validation.
Ensures:
1. Group independence (same scaffold always in same fold).
2. Stratification by Target (logS) and/or Temperature.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

# Suppress split warnings for small classes
warnings.filterwarnings("ignore", category=UserWarning)

def quantize_target(y, n_bins=5):
    """Bins continuous target into quantiles for stratification."""
    y = pd.to_numeric(y, errors='coerce')
    try:
        return pd.qcut(y, n_bins, labels=False, duplicates='drop').fillna(-1).astype(int)
    except ValueError:
        # Fallback to equal-width bins if quantiles fail (e.g. too many ties)
        return pd.cut(y, n_bins, labels=False, duplicates='drop').fillna(-1).astype(int)

def get_temp_bin(t, step=5):
    """Bins temperature."""
    t = pd.to_numeric(t, errors='coerce')
    return (np.round(t / step) * step).fillna(-999).astype(int)

def main():
    ap = argparse.ArgumentParser(description="Generate Group Stratified Folds.")
    ap.add_argument("--input", required=True, help="Train input file")
    ap.add_argument("--out", default="folds.parquet", help="Output file with 'fold' column")
    
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--group-col", default="cluster_ecfp4_0p7")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--temp-col", default="temp_C")
    
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    
    # Stratification params
    ap.add_argument("--strat-mode", choices=["target", "temp", "both"], default="both")
    ap.add_argument("--bins", type=int, default=5, help="Target bins")
    ap.add_argument("--temp-step", type=float, default=5.0, help="Temp bin step")

    args = ap.parse_args()

    # 1. Load
    print(f"[Folds] Loading {args.input}...")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # Check columns
    req_cols = [args.id_col, args.group_col, args.target]
    if args.strat_mode in ["temp", "both"]:
        req_cols.append(args.temp_col)
    
    for c in req_cols:
        if c not in df.columns:
            sys.exit(f"[ERROR] Missing column: {c}")

    # 2. Create Group Meta-Data
    # We need one row per group to perform the split
    grp = df[[args.group_col]].drop_duplicates().reset_index(drop=True)
    
    # Calculate stratification labels per group (aggregate median/mode)
    if args.strat_mode in ["target", "both"]:
        # Median target per group
        y_med = df.groupby(args.group_col)[args.target].median()
        grp["y_bin"] = grp[args.group_col].map(y_med).pipe(quantize_target, n_bins=args.bins)
        
    if args.strat_mode in ["temp", "both"]:
        # Mode temp per group (most frequent temp for this scaffold)
        # If scaffold has multiple temps, pick the most common one
        t_mode = df.groupby(args.group_col)[args.temp_col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else x.median())
        grp["t_bin"] = grp[args.group_col].map(t_mode).pipe(get_temp_bin, step=args.temp_step)

    # Combine labels
    if args.strat_mode == "target":
        grp["strata"] = grp["y_bin"].astype(str)
    elif args.strat_mode == "temp":
        grp["strata"] = grp["t_bin"].astype(str)
    else:
        grp["strata"] = grp["y_bin"].astype(str) + "|" + grp["t_bin"].astype(str)

    # 3. Perform Split
    print(f"[Folds] Splitting {len(grp)} groups into {args.n_splits} folds (Strategy: {args.strat_mode})...")
    
    X = grp[args.group_col].values
    y = grp["strata"].values
    
    # Check if stratification is possible (min 2 members per class)
    # If a class has fewer members than n_splits, StratifiedKFold warns/fails.
    # We filter extremely rare classes into 'OTHER' to avoid this.
    counts = grp["strata"].value_counts()
    rare_classes = counts[counts < args.n_splits].index
    y_safe = grp["strata"].mask(grp["strata"].isin(rare_classes), "OTHER").values

    try:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_gen = skf.split(X, y_safe)
    except Exception as e:
        print(f"[WARN] Stratified Split failed ({e}). Fallback to simple KFold.")
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_gen = kf.split(X)

    # Map Fold to Group
    group_to_fold = {}
    for fold_id, (_, test_idx) in enumerate(split_gen):
        test_groups = X[test_idx]
        for g in test_groups:
            group_to_fold[g] = fold_id

    # 4. Map Fold to Rows
    df["fold"] = df[args.group_col].map(group_to_fold)
    
    # Handle any unmapped rows (shouldn't happen, but for safety)
    if df["fold"].isna().any():
        n_nan = df["fold"].isna().sum()
        print(f"[WARN] {n_nan} rows have no fold assigned (missing group?). Assigning fold -1.")
        df["fold"] = df["fold"].fillna(-1)

    df["fold"] = df["fold"].astype(int)

    # 5. Save output (Lightweight: only ID, Group, Fold)
    out_df = df[[args.id_col, "fold", args.group_col]].copy()
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    
    # Stats
    print(out_df["fold"].value_counts().sort_index().to_string())
    print(f"[Folds] Saved to {args.out}")

if __name__ == "__main__":
    main()