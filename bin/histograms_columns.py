#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
histograms_columns.py
---------------------
Generates comparative histograms (Train vs Test).
Features:
- Auto-detection of Temperature units (K -> C).
- Overlay mode for distribution checks.
- Log1p transformation support.
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def pick_columns(df, cols_cli):
    wanted = set(cols_cli or [])
    existing = set(df.columns)
    
    # Find valid columns
    selected = sorted(list(wanted & existing))
    missing = sorted(list(wanted - existing))
    
    if missing:
        print(f"[WARN] Columns not found in dataset: {missing}", file=sys.stderr)
        
    return selected

def convert_temp_k_to_c(colname, arr):
    """Heuristic to convert Kelvin to Celsius for plotting."""
    lower = colname.lower()
    if "temp" in lower and (arr.mean() > 150): # Simple heuristic
        return arr - 273.15, f"{colname} (°C)"
    return arr, colname

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="hist_out")
    ap.add_argument("--cols", nargs="+", required=True)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--overlay", action="store_true", help="Overlay Train/Test")
    ap.add_argument("--round-step", type=float, default=None)
    
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_tr = pd.read_parquet(args.train)
    df_te = pd.read_parquet(args.test)

    cols = pick_columns(df_tr, args.cols)
    if not cols:
        print("[ERROR] No valid columns found to plot.")
        sys.exit(0)

    for c in cols:
        if c not in df_te.columns: continue
        
        s_tr = pd.to_numeric(df_tr[c], errors='coerce').dropna().values
        s_te = pd.to_numeric(df_te[c], errors='coerce').dropna().values
        
        # Convert Temp if needed
        s_tr, label_tr = convert_temp_k_to_c(c, s_tr)
        s_te, label_te = convert_temp_k_to_c(c, s_te)
        xlabel = label_tr

        # Stats
        print(f"[{c}] Train: {len(s_tr)}, Test: {len(s_te)}")

        plt.figure(figsize=(8, 5))
        
        # Common range
        all_data = np.concatenate([s_tr, s_te])
        if len(all_data) == 0: continue
        
        xmin, xmax = all_data.min(), all_data.max()
        
        if args.overlay:
            plt.hist(s_tr, bins=args.bins, range=(xmin, xmax), alpha=0.5, density=True, label="Train", color='blue')
            plt.hist(s_te, bins=args.bins, range=(xmin, xmax), alpha=0.5, density=True, label="Test", color='orange')
            plt.ylabel("Density")
            plt.legend()
        else:
            plt.hist(all_data, bins=args.bins, color='gray', alpha=0.7)
            plt.ylabel("Count")

        plt.title(f"Distribution: {c}")
        plt.xlabel(xlabel)
        plt.grid(True, alpha=0.3)
        
        out_path = os.path.join(args.outdir, f"hist_{c}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()