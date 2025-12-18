#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stratified_split.py (Single Split)
----------------------------------
Generates ONE Train/Test split ensuring:
1. Scaffold splitting (Group separation).
2. Stratification (LogS/Temp balance).
3. 80/20 Ratio preservation.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

# ---------------- UTILS (Iguales que antes) ----------------

def build_strat_label(df, temp_col="temp_C", temp_step=5, target_col="logS", n_bins=5):
    if temp_col in df.columns:
        t = pd.to_numeric(df[temp_col], errors="coerce")
        t_bin = (np.round(t / temp_step) * temp_step).fillna(-999).astype(int).astype(str)
    else:
        t_bin = "NoTemp"

    if target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce")
        try:
            y_bin = pd.qcut(y, q=n_bins, labels=False, duplicates='drop').fillna(-1).astype(int).astype(str)
        except:
            y_bin = "All"
    else:
        y_bin = "NoTarget"

    return t_bin + "|" + y_bin

def pick_group_label(labels):
    c = Counter(labels)
    return c.most_common(1)[0][0]

def collapse_rare_classes(group_df, label_col="group_strat_label", min_groups=2):
    counts = group_df[label_col].value_counts()
    rare_labels = counts[counts < min_groups].index
    return group_df[label_col].mask(group_df[label_col].isin(rare_labels), "OTHER")

def smart_balanced_split_single(grp_meta, test_size, seed):
    """
    Versión simplificada para UN solo split.
    """
    rng = np.random.default_rng(seed)
    strata = grp_meta["group_strat_label"].unique()

    train_groups_all = []
    test_groups_all = []
    
    for s in strata:
        subset = grp_meta[grp_meta["group_strat_label"] == s]
        indices = subset.index.values
        sizes = subset["size"].values
        
        total_rows = sizes.sum()
        target_test_rows = int(total_rows * test_size)
        
        # Barajamos
        perm = rng.permutation(len(indices))
        indices = indices[perm]
        sizes = sizes[perm]
        
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
                if current_test_rows == 0: # Forzar al menos uno si está vacío
                    test_g_idxs.append(g_idx)
                    current_test_rows += sz
                else:
                    train_g_idxs.append(g_idx)
        
        train_groups_all.extend(train_g_idxs)
        test_groups_all.extend(test_g_idxs)
        
    return train_groups_all, test_groups_all

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--group-col", default="cluster_ecfp4_0p7")
    ap.add_argument("--temp-col", default="temp_C")
    ap.add_argument("--target-col", default="logS")
    ap.add_argument("--temp-step", type=float, default=5.0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-groups-per-class", type=int, default=2)
    ap.add_argument("--outdir", default=".")
    
    args = ap.parse_args()
    
    print(f"[Split] Leyendo {args.input}...")
    df = pd.read_parquet(args.input)

    # Crear etiquetas de estratificación
    df["strat_label"] = build_strat_label(df, args.temp_col, args.temp_step, args.target_col)
    
    # Agrupar por scaffold
    grp_stats = df.groupby(args.group_col)["strat_label"].agg(['count', lambda x: pick_group_label(list(x))])
    grp_stats.columns = ["size", "group_strat_label"]
    grp_stats = grp_stats.reset_index()
    grp_stats["group_strat_label"] = collapse_rare_classes(grp_stats, "group_strat_label", args.min_groups_per_class)

    print(f"[Split] Generando partición 80/20 (Seed={args.seed})...")
    train_idx_g, test_idx_g = smart_balanced_split_single(grp_stats, args.test_size, args.seed)

    # Filtrar DF original
    train_grps = set(grp_stats.iloc[train_idx_g][args.group_col])
    test_grps  = set(grp_stats.iloc[test_idx_g][args.group_col])
    
    mask_train = df[args.group_col].isin(train_grps)
    mask_test  = df[args.group_col].isin(test_grps)
    
    df_train = df[mask_train].drop(columns=["strat_label"])
    df_test  = df[mask_test].drop(columns=["strat_label"])
    
    # Guardar directamente como train.parquet y test.parquet
    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df_train.to_parquet(out_path / "train.parquet", index=False)
    df_test.to_parquet(out_path / "test.parquet", index=False)
    
    n_tr, n_te = len(df_train), len(df_test)
    ratio = n_te / (n_tr + n_te) if (n_tr + n_te) > 0 else 0
    print(f"[Split] Completado: Train={n_tr}, Test={n_te} (Test Ratio: {ratio:.1%})")

if __name__ == "__main__":
    main()