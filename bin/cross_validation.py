#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross Validation: CV Results Aggregation and Consensus Calculation
===================================================================

Aggregates results from multiple cross-validation splits to compute consensus
metrics and predictions. Provides robust performance estimates that average
over fold variability.

Consensus Strategy:
    For molecules appearing in multiple CV splits:
    1. Collect all predictions from different folds
    2. Average predictions to get consensus estimate
    3. Calculate metrics on consensus predictions
    
    This reduces variance and provides more stable estimates.

Arguments:
    --inputs       : Metrics JSON files from each CV split
    --level0-files : Level-0 prediction parquet files
    --blend-files  : Blend prediction parquet files
    --stack-files  : Stack prediction parquet files (optional)
    --out-csv      : Output CSV filename (default: montecarlo_summary.csv)
    --outdir       : Output directory

Usage:
    python cross_validation.py \\
        --inputs metrics_*.json \\
        --level0-files level0_*.parquet \\
        --blend-files blend_*.parquet \\
        --stack-files stack_*.parquet \\
        --outdir cv_out

Output:
    - montecarlo_summary.csv: Summary statistics across splits
    - test_level0.parquet: Consensus Level-0 predictions
    - test_blend.parquet: Consensus blended predictions
    - test_stack.parquet: Consensus stacked predictions
    - metrics_cv_consensus.json: Metrics on consensus predictions
    - best_strategy.txt: Winner (blend or stack)

Notes:
    - Missing files are skipped with warnings
    - Consensus calculated by averaging predictions per molecule
    - Physiological range metrics (35-38°C) included when temperature available
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================================
# UTILITIES
# ============================================================================

def calc_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"rmse": np.nan, "r2": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "r2": float(r2_score(y_true[mask], y_pred[mask]))
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for CV aggregation."""
    
    ap = argparse.ArgumentParser(
        description="Aggregate cross-validation results and compute consensus metrics."
    )
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Metrics JSON files from each split")
    ap.add_argument("--level0-files", nargs="+", required=True,
                    help="Level-0 prediction parquet files")
    ap.add_argument("--blend-files", nargs="+", required=True,
                    help="Blend prediction parquet files")
    ap.add_argument("--stack-files", nargs="+", default=[],
                    help="Stack prediction parquet files (optional)")
    ap.add_argument("--out-csv", default="montecarlo_summary.csv",
                    help="Output CSV filename")
    ap.add_argument("--outdir", default=".",
                    help="Output directory")
    
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    # -------------------------------------------------------------------------
    # Aggregate Metrics from JSON Files
    # -------------------------------------------------------------------------
    # Model columns (updated: tpsa → physics)
    MODELS = ["y_xgb", "y_lgbm", "y_cat", "y_chemprop", "y_physics", "y_pred_blend", "y_pred_stack"]
    
    data_global = {
        m: {"test_rmse": [], "test_r2": [], "train_rmse": [], "train_r2": []}
        for m in MODELS
    }
    
    valid_files = 0
    for fpath in args.inputs:
        try:
            with open(fpath, 'r') as f:
                d = json.load(f)
            
            for m in MODELS:
                if m in d:
                    # Test metrics
                    if "test_rmse" in d[m]:
                        data_global[m]["test_rmse"].append(d[m]["test_rmse"])
                        data_global[m]["test_r2"].append(d[m]["test_r2"])
                    elif "rmse" in d[m]:
                        data_global[m]["test_rmse"].append(d[m]["rmse"])
                        data_global[m]["test_r2"].append(d[m]["r2"])
                    
                    # Train metrics
                    if "train_rmse" in d[m]:
                        data_global[m]["train_rmse"].append(d[m]["train_rmse"])
                        data_global[m]["train_r2"].append(d[m]["train_r2"])
            
            valid_files += 1
        except Exception as e:
            print(f"[WARN] Error reading {fpath}: {e}")
    
    print(f"[CV] Loaded metrics from {valid_files} files")
    
    # -------------------------------------------------------------------------
    # Build Summary Statistics
    # -------------------------------------------------------------------------
    rows = []
    for m in MODELS:
        te_rmse = np.array(data_global[m]["test_rmse"])
        te_r2 = np.array(data_global[m]["test_r2"])
        tr_rmse = np.array(data_global[m]["train_rmse"])
        tr_r2 = np.array(data_global[m]["train_r2"])
        
        if len(te_rmse) == 0:
            continue
        
        rows.append({
            "Model": m,
            "Test_RMSE": np.mean(te_rmse),
            "Test_RMSE_Std": np.std(te_rmse),
            "Test_R2": np.mean(te_r2),
            "Test_R2_Std": np.std(te_r2),
            "Train_RMSE": np.mean(tr_rmse) if len(tr_rmse) > 0 else None,
            "Train_R2": np.mean(tr_r2) if len(tr_r2) > 0 else None
        })
    
    df_res = pd.DataFrame(rows)
    
    if not df_res.empty:
        df_res = df_res.sort_values("Test_RMSE")
        df_res.to_csv(outdir / args.out_csv, index=False)
        
        # Determine winner
        winner = df_res.iloc[0]
        best_model = winner['Model'].replace("y_pred_", "").replace("y_", "")
        if best_model not in ["blend", "stack"]:
            best_model = "blend"  # Default to blend if base model wins
    else:
        best_model = "blend"
    
    # Save best strategy
    with open("best_strategy.txt", "w") as f:
        f.write(best_model)
    
    print(f"[CV] Best strategy: {best_model}")
    
    # -------------------------------------------------------------------------
    # Merge and Compute Consensus Predictions
    # -------------------------------------------------------------------------
    print(f"[CV] Merging data from {len(args.level0_files)} splits...")
    
    l0_files = sorted(args.level0_files)
    bl_files = sorted(args.blend_files)
    st_files = sorted(args.stack_files) if args.stack_files else [None] * len(l0_files)
    
    dfs = []
    for f0, fb, fs in zip(l0_files, bl_files, st_files):
        try:
            d0 = pd.read_parquet(f0)
            db = pd.read_parquet(fb)
            
            # Merge Level-0 with Blend
            m = d0.merge(db[["row_uid", "y_pred_blend"]], on="row_uid", how="left")
            
            # Merge Stack if available
            if fs and Path(fs).exists():
                ds = pd.read_parquet(fs)
                m = m.merge(ds[["row_uid", "y_pred_stack"]], on="row_uid", how="left")
            
            dfs.append(m)
        except Exception as e:
            print(f"[WARN] Error processing files: {e}")
    
    if not dfs:
        print("[WARN] No data merged. Creating empty outputs.")
        return
    
    # Concatenate all splits
    big = pd.concat(dfs, ignore_index=True)
    
    # -------------------------------------------------------------------------
    # Calculate Consensus (Average per Molecule)
    # -------------------------------------------------------------------------
    num_cols = big.select_dtypes(include=[np.number]).columns.tolist()
    if "fold" in num_cols:
        num_cols.remove("fold")
    
    df_mean = big.groupby("row_uid")[num_cols].mean().reset_index()
    print(f"[CV] Unique molecules (consensus): {len(df_mean):,}")
    
    # -------------------------------------------------------------------------
    # Save Consensus Outputs
    # -------------------------------------------------------------------------
    # Level-0 predictions
    l0_cols = [c for c in df_mean.columns if c.startswith("y_") and "blend" not in c and "stack" not in c]
    cols_out = ["row_uid"]
    if "logS" in df_mean.columns:
        cols_out.append("logS")
    if "temp_C" in df_mean.columns:
        cols_out.append("temp_C")
    cols_out += l0_cols
    
    df_mean[cols_out].to_parquet(outdir / "test_level0.parquet", index=False)
    
    # Blend predictions
    if "y_pred_blend" in df_mean.columns:
        df_mean[["row_uid", "y_pred_blend"]].to_parquet(outdir / "test_blend.parquet", index=False)
    
    # Stack predictions
    if "y_pred_stack" in df_mean.columns:
        df_mean[["row_uid", "y_pred_stack"]].to_parquet(outdir / "test_stack.parquet", index=False)
    
    # -------------------------------------------------------------------------
    # Calculate Consensus Metrics
    # -------------------------------------------------------------------------
    metrics_consensus = {}
    
    if "logS" in df_mean.columns:
        y_true = df_mean["logS"].values
        
        # Metrics on consensus predictions
        for m in MODELS:
            if m in df_mean.columns:
                metrics_consensus[m] = calc_metrics(y_true, df_mean[m].values)
        
        # Physiological range metrics
        if "temp_C" in df_mean.columns:
            t_val = df_mean["temp_C"].values
            mask_phys = (t_val >= 35) & (t_val <= 38)
            
            if mask_phys.any():
                metrics_consensus["physio_range"] = {}
                for m in MODELS:
                    if m in df_mean.columns:
                        metrics_consensus["physio_range"][m] = calc_metrics(
                            y_true[mask_phys],
                            df_mean.loc[mask_phys, m].values
                        )
    
    # Save consensus metrics
    json_path = outdir / "metrics_cv_consensus.json"
    with open(json_path, "w") as f:
        json.dump(metrics_consensus, f, indent=2)
    
    print(f"[CV] Consensus metrics saved to: {json_path}")


if __name__ == "__main__":
    main()
