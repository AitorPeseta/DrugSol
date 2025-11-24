#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stratified_split.py
-------------------
Splits data into Train/Test sets ensuring:
1. No data leakage: Split is done by MOLECULAR GROUP (e.g. Scaffold/Cluster).
2. Balanced conditions: Split is STRATIFIED by Solvent + Temperature bin.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# ---------------- UTILS ----------------

def build_strat_label(df, solvent_col="solvent", temp_col="temp_C", temp_step=5):
    """Creates a stratification label: '<solvent>|<temp_bin>'."""
    if solvent_col not in df.columns or temp_col not in df.columns:
        raise ValueError(f"Missing columns: {solvent_col} or {temp_col}")

    # Bin temperature
    t = pd.to_numeric(df[temp_col], errors="coerce")
    t_bin = (np.round(t / temp_step) * temp_step).fillna(-999)
    
    # Format bin as integer if possible for cleaner labels
    t_str = t_bin.apply(lambda x: str(int(x)) if x == int(x) else str(x))
    
    return df[solvent_col].astype(str) + "|" + t_str

def pick_group_label(labels):
    """
    Returns a representative label for a group (Majority Vote).
    If a scaffold appears mostly in 'Water|25', that's its label.
    """
    c = Counter(labels)
    # Find most common label
    max_freq = c.most_common(1)[0][1]
    candidates = sorted([k for k, v in c.items() if v == max_freq])
    return candidates[0]

def collapse_rare_classes(group_df, label_col="group_strat_label", min_groups=2):
    """
    Collapses rare stratification classes into 'OTHER'.
    Classes with fewer than `min_groups` cannot be stratified effectively.
    """
    counts = group_df[label_col].value_counts()
    rare_labels = counts[counts < min_groups].index
    
    # Replace rare labels with 'OTHER'
    collapsed = group_df[label_col].mask(group_df[label_col].isin(rare_labels), "OTHER")
    return collapsed

# ---------------- MAIN ----------------

def main():
    ap = argparse.ArgumentParser(description="Group Stratified Split.")
    ap.add_argument("--input", "-i", required=True, help="Input Parquet/CSV with group column.")
    ap.add_argument("--group-col", required=True, help="Column to group by (e.g., cluster_ecfp4_0p7).")
    ap.add_argument("--temp-col", default="temp_C", help="Temperature column.")
    ap.add_argument("--temp-step", type=float, default=5.0, help="Bin size for temperature stratification.")
    ap.add_argument("--test-size", "-t", type=float, default=0.2, help="Fraction of groups in Test set.")
    ap.add_argument("--seed", "-s", type=int, default=42, help="Random seed.")
    ap.add_argument("--min-groups-per-class", type=int, default=2, help="Min groups required to keep a class separate.")
    ap.add_argument("--invalid-group", type=int, default=-1, help="Group ID to exclude (e.g., unclustered noise).")
    ap.add_argument("--save-csv", action="store_true", help="Save CSV copies.")
    
    args = ap.parse_args()

    # 1. Load Data
    print(f"[Split] Loading {args.input}...")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # 2. Filter Invalid Groups (Optional)
    # e.g., Cluster -1 usually means 'outlier/noise' in clustering algorithms
    if args.invalid_group is not None and args.group_col in df.columns:
        mask = df[args.group_col] != args.invalid_group
        n_excl = (~mask).sum()
        if n_excl > 0:
            print(f"[Split] Excluding {n_excl} rows with group {args.invalid_group}.")
            df = df[mask].copy()

    if df.empty:
        sys.exit("[ERROR] No data left after filtering.")

    # 3. Create Stratification Labels (Per Row)
    df["strat_label"] = build_strat_label(df, args.solvent_col if 'solvent_col' in args else 'solvent', args.temp_col, args.temp_step)

    # 4. Assign Label to Group (Per Group)
    # We split Groups, not Rows. So each Group needs 1 label.
    grp_meta = df.groupby(args.group_col)["strat_label"].apply(list).reset_index()
    grp_meta["group_strat_label"] = grp_meta["strat_label"].apply(pick_group_label)

    # 5. Handle Rare Classes
    # If 'Methanol|50C' has only 1 scaffold, we can't put it in both Train and Test. Move to 'OTHER'.
    grp_meta["group_strat_label"] = collapse_rare_classes(
        grp_meta, "group_strat_label", args.min_groups_per_class
    )

    groups = grp_meta[args.group_col].values
    labels = grp_meta["group_strat_label"].values
    
    n_classes = len(set(labels))
    print(f"[Split] Unique Groups: {len(groups)} | Stratification Classes: {n_classes}")

    # 6. Perform Split
    indices = np.arange(len(groups))
    
    if n_classes > 1:
        print("[Split] Using Stratified Shuffle Split (Balancing conditions)...")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        try:
            train_idx, test_idx = next(splitter.split(indices, labels))
        except ValueError as e:
            print(f"[WARN] Stratification failed ({e}). Falling back to random group split.")
            splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
            train_idx, test_idx = next(splitter.split(indices, groups=groups))
    else:
        print("[Split] Only 1 class found. Using Random Group Split...")
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(indices, groups=groups))

    # 7. Map back to Rows
    train_groups = set(groups[train_idx])
    test_groups  = set(groups[test_idx])
    
    # Verify leakage
    assert len(train_groups & test_groups) == 0, "CRITICAL: Data leakage detected (shared groups)."

    mask_train = df[args.group_col].isin(train_groups)
    mask_test  = df[args.group_col].isin(test_groups)

    df_train = df[mask_train].drop(columns=["strat_label"])
    df_test  = df[mask_test].drop(columns=["strat_label"])

    # 8. Save
    df_train.to_parquet("train.parquet", index=False)
    df_test.to_parquet("test.parquet", index=False)

    if args.save_csv:
        df_train.to_csv("train.csv", index=False)
        df_test.to_csv("test.csv", index=False)

    # 9. Stats
    print(f"[Split] TRAIN: {len(df_train)} rows ({len(train_groups)} groups)")
    print(f"[Split] TEST : {len(df_test)} rows ({len(test_groups)} groups)")

if __name__ == "__main__":
    main()