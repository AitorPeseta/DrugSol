#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stratified_split.py (Smart Balanced Monte Carlo)
------------------------------------------------
Generates N independent Train/Test splits ensuring:
1. Scaffold splitting (GroupShuffleSplit).
2. Stratification (LogS/Temp balance).
3. SMART SIZE BALANCE: Minimizes deviation from target ratio (80/20).
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

# ---------------- UTILS ----------------

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

def smart_balanced_split(grp_meta, n_splits, test_size, seed):
    """
    Algoritmo Smart Fit:
    Baraja los grupos y los asigna al Test SOLO si eso reduce el error
    respecto al tamaño objetivo. Evita "pasarse de frenada" con grupos grandes.
    """
    rng = np.random.default_rng(seed)
    splits = []
    strata = grp_meta["group_strat_label"].unique()

    for i in range(n_splits):
        iter_rng = np.random.default_rng(seed + i)
        
        train_groups_all = []
        test_groups_all = []
        
        for s in strata:
            subset = grp_meta[grp_meta["group_strat_label"] == s]
            indices = subset.index.values
            sizes = subset["size"].values
            
            # Objetivo exacto para este estrato
            total_rows = sizes.sum()
            target_test_rows = int(total_rows * test_size)
            
            # Barajamos
            perm = iter_rng.permutation(len(indices))
            indices = indices[perm]
            sizes = sizes[perm]
            
            current_test_rows = 0
            test_g_idxs = []
            train_g_idxs = []
            
            for g_idx, sz in zip(indices, sizes):
                # CRITERIO SMART:
                # Calculamos el error (distancia al objetivo) si lo añadimos vs si no lo añadimos
                dist_if_add = abs((current_test_rows + sz) - target_test_rows)
                dist_if_skip = abs(current_test_rows - target_test_rows)
                
                # Si añadirlo nos acerca al objetivo (o nos deja igual), lo metemos.
                # Si añadirlo nos aleja (nos pasamos mucho), lo mandamos a Train.
                if dist_if_add < dist_if_skip:
                    test_g_idxs.append(g_idx)
                    current_test_rows += sz
                else:
                    # Caso especial: Si el test está vacío, estamos obligados a meter al menos uno
                    if current_test_rows == 0:
                        test_g_idxs.append(g_idx)
                        current_test_rows += sz
                    else:
                        train_g_idxs.append(g_idx)
            
            train_groups_all.extend(train_g_idxs)
            test_groups_all.extend(test_g_idxs)
            
        splits.append((train_groups_all, test_groups_all))
        
    return splits

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
    ap.add_argument("--n-splits", type=int, default=1)
    ap.add_argument("--min-groups-per-class", type=int, default=2)
    ap.add_argument("--outdir", default=".")
    
    args = ap.parse_args()
    
    print(f"[Split] Loading {args.input}...")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    print(f"[Split] Estratificando por {args.temp_col} y {args.target_col}...")
    df["strat_label"] = build_strat_label(df, args.temp_col, args.temp_step, args.target_col)
    
    grp_stats = df.groupby(args.group_col)["strat_label"].agg(['count', lambda x: pick_group_label(list(x))])
    grp_stats.columns = ["size", "group_strat_label"]
    grp_stats = grp_stats.reset_index()
    grp_stats["group_strat_label"] = collapse_rare_classes(grp_stats, "group_strat_label", args.min_groups_per_class)

    print(f"[Split] Generando {args.n_splits} splits SMART (Target: {args.test_size:.0%})...")
    split_indices = smart_balanced_split(grp_stats, args.n_splits, args.test_size, args.seed)

    for i, (train_idx_g, test_idx_g) in enumerate(split_indices):
        iter_name = f"split_{i+1}"
        out_folder = Path(args.outdir) / iter_name
        out_folder.mkdir(parents=True, exist_ok=True)
        
        train_grps = set(grp_stats.iloc[train_idx_g][args.group_col])
        test_grps  = set(grp_stats.iloc[test_idx_g][args.group_col])
        
        mask_train = df[args.group_col].isin(train_grps)
        mask_test  = df[args.group_col].isin(test_grps)
        
        df_train = df[mask_train].drop(columns=["strat_label"])
        df_test  = df[mask_test].drop(columns=["strat_label"])
        
        df_train.to_parquet(out_folder / "train.parquet", index=False)
        df_test.to_parquet(out_folder / "test.parquet", index=False)
        
        n_tr, n_te = len(df_train), len(df_test)
        ratio = n_te / (n_tr + n_te) if (n_tr + n_te) > 0 else 0
        print(f"  > {iter_name}: Train={n_tr}, Test={n_te} (Test Ratio: {ratio:.1%})")

    print("[Split] Finalizado.")

if __name__ == "__main__":
    main()