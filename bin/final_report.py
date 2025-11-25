#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_report.py
---------------
Generates comprehensive performance reports.
STRUCTURE:
  - Global Analysis (All temperatures)
  - Physiological Analysis (35-38 C zoom)
Outputs plots and JSON metrics for both scopes.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Style configuration
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = {
    "y_xgb": "#e67e22",      # Orange
    "y_lgbm": "#f1c40f",     # Yellow
    "y_chemprop": "#9b59b6", # Purple
    "y_tpsa": "#95a5a6",     # Gray
    "y_pred_blend": "#2ecc71", # Green (Main)
    "y_pred_stack": "#3498db"  # Blue
}

# ---------------- METRICS & PLOTTING ENGINE ----------------

def calculate_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return {"rmse": None, "r2": None, "n": 0}
    y_t, y_p = y_true[mask], y_pred[mask]
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "r2": float(r2_score(y_t, y_p)),
        "n": int(mask.sum())
    }

def plot_scatter(df, true_col, pred_col, model_name, out_file):
    plt.figure(figsize=(7, 6))
    
    y_t = df[true_col]
    y_p = df[pred_col]
    m = calculate_metrics(y_t, y_p)
    
    # Plot
    color = PALETTE.get(pred_col, "#34495e")
    sns.scatterplot(x=y_t, y=y_p, alpha=0.5, edgecolor="w", s=40, color=color)
    
    # Ideal line
    lims = [min(y_t.min(), y_p.min()) - 0.5, max(y_t.max(), y_p.max()) + 0.5]
    plt.plot(lims, lims, 'k--', alpha=0.7, zorder=0)
    
    # Regression line
    sns.regplot(x=y_t, y=y_p, scatter=False, color="red", line_kws={'linewidth': 1.5, 'alpha': 0.8})
    
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Experimental LogS")
    plt.ylabel(f"Predicted LogS ({model_name})")
    plt.title(f"{model_name}\nRMSE: {m['rmse']:.3f} | $R^2$: {m['r2']:.3f} | N: {m['n']}")
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    return m

def plot_residuals_hist(df, true_col, pred_col, model_name, out_file, bin_step):
    plt.figure(figsize=(8, 5))
    
    res = df[pred_col] - df[true_col]
    res = res.dropna()
    
    # Dynamic bins centered at 0
    max_val = np.abs(res).max()
    max_val = np.ceil(max_val)
    bins = np.arange(-max_val, max_val + bin_step, bin_step)
    
    color = PALETTE.get(pred_col, "#2ecc71")
    sns.histplot(res, bins=bins, kde=True, color=color, edgecolor="white", alpha=0.6)
    plt.axvline(0, color='k', linestyle='--', linewidth=1.5)
    
    plt.xlabel("Error (Predicted - Experimental)")
    plt.ylabel("Count")
    plt.title(f"{model_name} Residuals\n(Bin Size: {bin_step})")
    plt.xlim(-3, 3) # Focus on relevant area
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

def plot_bland_altman(df, true_col, pred_col, model_name, out_file):
    plt.figure(figsize=(7, 6))
    res = df[pred_col] - df[true_col]
    
    color = PALETTE.get(pred_col, "#9b59b6")
    sns.scatterplot(x=df[true_col], y=res, alpha=0.5, color=color, edgecolor="w")
    plt.axhline(0, color='k', linestyle='--', linewidth=1.5)
    
    # Confidence interval 2SD
    sd = res.std()
    plt.axhline(1.96*sd, color='r', linestyle=':', alpha=0.5)
    plt.axhline(-1.96*sd, color='r', linestyle=':', alpha=0.5)
    
    plt.xlabel("Experimental LogS")
    plt.ylabel("Residual (Pred - Exp)")
    plt.title(f"{model_name} Bias Check")
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

def generate_suite(df, target_col, models_map, out_folder, bin_step):
    """Runs all plots for a given dataframe subset."""
    metrics = {}
    out_folder.mkdir(parents=True, exist_ok=True)
    
    for col, name in models_map.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        
        # 1. Scatter
        m = plot_scatter(df, target_col, col, name, out_folder / f"scatter_{safe_name}.png")
        metrics[name] = m
        
        # 2. Histogram
        plot_residuals_hist(df, target_col, col, name, out_folder / f"hist_{safe_name}.png", bin_step)
        
        # 3. Bias
        plot_bland_altman(df, target_col, col, name, out_folder / f"bias_{safe_name}.png")
        
    return metrics

# ---------------- MAIN ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level0", required=True)
    ap.add_argument("--blend", required=True)
    ap.add_argument("--stack", default=None)
    ap.add_argument("--metadata", required=True, help="Original Test parquet with temp_C")
    
    ap.add_argument("--target", default="logS")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--temp-col", default="temp_C")
    ap.add_argument("--bin-step", type=float, default=0.2)
    
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    root = Path(args.outdir)
    
    # 1. Load & Merge Everything
    print("[Report] Loading data...")
    df_l0 = pd.read_parquet(args.level0)
    df_blend = pd.read_parquet(args.blend)
    df_meta = pd.read_parquet(args.metadata)
    
    # Ensure IDs match (string cleanup)
    for d in [df_l0, df_blend, df_meta]:
        if args.id_col in d.columns:
            d[args.id_col] = d[args.id_col].astype(str).str.strip()

    # Start merging
    # Base: predictions
    df = df_l0.merge(df_blend[[args.id_col, "y_pred_blend"]], on=args.id_col, how="left")
    
    if args.stack and Path(args.stack).exists():
        df_stack = pd.read_parquet(args.stack)
        if args.id_col in df_stack.columns:
            df_stack[args.id_col] = df_stack[args.id_col].astype(str).str.strip()
        df = df.merge(df_stack[[args.id_col, "y_pred_stack"]], on=args.id_col, how="left")

    # Add Metadata (Temp, QED, etc if available)
    # We need 'temp_C' for filtering
    cols_meta = [args.id_col, args.temp_col]
    if args.target not in df.columns and args.target in df_meta.columns:
        cols_meta.append(args.target)
        
    df = df.merge(df_meta[cols_meta], on=args.id_col, how="left")
    
    print(f"[Report] Master DF shape: {df.shape}")
    
    # Define Models to Analyze
    possible_models = {
        "y_xgb": "XGBoost",
        "y_lgbm": "LightGBM",
        "y_chemprop": "GNN (Chemprop)",
        "y_tpsa": "Baseline (TPSA)",
        "y_pred_stack": "Ensemble (Stack)",
        "y_pred_blend": "Ensemble (Blend)"
    }
    # Filter existing columns
    models_map = {k: v for k, v in possible_models.items() if k in df.columns}

    # --- RUN 1: GLOBAL (All Data) ---
    print(f"[Report] Generating GLOBAL plots (N={len(df)})...")
    metrics_global = generate_suite(df, args.target, models_map, root / "plots_global", args.bin_step)
    (root / "metrics_global.json").write_text(json.dumps(metrics_global, indent=2))

    # --- RUN 2: PHYSIOLOGICAL (35-38 C) ---
    if args.temp_col in df.columns:
        # Convert to numeric just in case
        temp_vals = pd.to_numeric(df[args.temp_col], errors='coerce')
        mask_physio = (temp_vals >= 35) & (temp_vals <= 38)
        df_physio = df[mask_physio].copy()
        
        print(f"[Report] Generating PHYSIO plots (N={len(df_physio)})...")
        if len(df_physio) > 10:
            metrics_physio = generate_suite(df_physio, args.target, models_map, root / "plots_physio", args.bin_step)
            (root / "metrics_physio.json").write_text(json.dumps(metrics_physio, indent=2))
        else:
            print("[WARN] Not enough physiological data points to plot (<10).")
    else:
        print(f"[WARN] Temperature column '{args.temp_col}' not found. Skipping physio plots.")

    # Generate Summary CSV
    summary = []
    for model, res in metrics_global.items():
        res["scope"] = "Global"
        res["model"] = model
        summary.append(res)
        
    if 'metrics_physio' in locals():
        for model, res in metrics_physio.items():
            res["scope"] = "Physio (35-38C)"
            res["model"] = model
            summary.append(res)
            
    pd.DataFrame(summary).to_csv(root / "metrics_summary.csv", index=False)
    print("[Report] Done.")

if __name__ == "__main__":
    main()