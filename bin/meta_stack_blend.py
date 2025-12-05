#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meta_stack_blend.py
-------------------
Meta-learning module.
Combines Out-of-Fold predictions from multiple models.
Strategies:
1. Stacking (Ridge Regression)
2. Blending (Linear combination with non-negative weights)
"""

import argparse
import json
import pickle
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- METRICS ----------------

def root_mse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

def r2(y, yhat):
    return float(r2_score(y, yhat))

# ---------------- UTILS ----------------

def read_any(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path}")

def collapse_oof(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates duplicates by (id, fold).
    y_true -> first, y_pred -> mean.
    """
    return df.groupby(["id", "fold"], as_index=False).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean")
    )

def merge_oofs(oof_files, labels):
    """
    Merges multiple OOF dataframes on [id, fold].
    Renames y_pred columns to oof_<label>.
    Returns: y (target), Z (matrix of preds), folds (array), merged_df
    """
    merged = None
    final_labels = []
    
    for i, f in enumerate(oof_files):
        df = read_any(f)
        label = labels[i] if labels and i < len(labels) else f"model_{i}"
        final_labels.append(label)
        
        # Ensure columns exist
        req = {"id", "fold", "y_true", "y_pred"}
        if not req.issubset(df.columns):
            raise ValueError(f"File {f} missing columns. Required: {req}")
            
        # Deduplicate if needed
        if df.duplicated(subset=["id", "fold"]).any():
            df = collapse_oof(df)
            
        # Prepare for merge
        df_renamed = df[["id", "fold", "y_true", "y_pred"]].rename(
            columns={"y_true": f"y_true_{label}", "y_pred": f"oof_{label}"}
        )
        
        if merged is None:
            merged = df_renamed
        else:
            merged = merged.merge(df_renamed, on=["id", "fold"], how="inner")
            
    if merged is None or merged.empty:
        raise ValueError("Merge resulted in empty dataframe (no overlapping IDs?).")
        
    # Extract Target (y) and Features (Z)
    # y should be consistent across models
    y_col = f"y_true_{final_labels[0]}"
    y = merged[y_col].values
    Z = merged[[f"oof_{lab}" for lab in final_labels]].values
    folds = merged["fold"].values
    
    return y, Z, folds, merged, final_labels

# ---------------- BLENDING (Simplex) ----------------

def simplex_project(v):
    """Projects vector v onto the probability simplex (sum=1, non-negative)."""
    v = np.array(v, dtype=float)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(n) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

def blend_weights(Z, y, l2=1e-8):
    """Finds optimal weights w for y ~ Z @ w, subject to w>=0, sum(w)=1."""
    Z = np.array(Z, dtype=float)
    y = np.array(y, dtype=float)
    
    # Least Squares solution (Ridge)
    n_models = Z.shape[1]
    A = Z.T @ Z + l2 * np.eye(n_models)
    b = Z.T @ y
    try:
        w_unconstrained = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_unconstrained = np.linalg.lstsq(A, b, rcond=None)[0]
        
    # Project to Simplex
    return simplex_project(w_unconstrained)

def blending_cv(Z, y, folds):
    """Out-of-Fold Blending."""
    yhat = np.zeros_like(y)
    unique_folds = np.unique(folds)
    
    for k in unique_folds:
        val_idx = (folds == k)
        trn_idx = ~val_idx
        
        # Learn weights on Train part of folds
        w_k = blend_weights(Z[trn_idx], y[trn_idx])
        # Apply to Val part
        yhat[val_idx] = Z[val_idx] @ w_k
        
    # Final weights on full data
    w_full = blend_weights(Z, y)
    return yhat, w_full

# ---------------- STACKING (Ridge) ----------------

def stacking_cv(Z, y, folds, seed=42):
    """Out-of-Fold Stacking with Ridge Regression."""
    yhat = np.zeros_like(y)
    unique_folds = np.unique(folds)
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Find best alpha globally first (simplification)
    ridge_cv = RidgeCV(alphas=alphas, cv=KFold(5, shuffle=True, random_state=seed))
    ridge_cv.fit(Z, y)
    best_alpha = ridge_cv.alpha_
    
    # OOF Predictions
    for k in unique_folds:
        val_idx = (folds == k)
        trn_idx = ~val_idx
        
        model_k = Ridge(alpha=best_alpha, random_state=seed)
        model_k.fit(Z[trn_idx], y[trn_idx])
        yhat[val_idx] = model_k.predict(Z[val_idx])
        
    # Final model on full data
    meta_full = Ridge(alpha=best_alpha, random_state=seed)
    meta_full.fit(Z, y)
    
    return yhat, meta_full, best_alpha

# ---------------- MAIN ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-common", nargs="+", required=True, help="List of OOF parquet files")
    ap.add_argument("--labels", nargs="+", help="Labels for models (lgbm xgb ...)")
    ap.add_argument("--metric", choices=["rmse", "r2"], default="rmse")
    ap.add_argument("--save-dir", default="meta_results")
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    
    outdir = Path(args.save_dir)
    (outdir / "stack").mkdir(parents=True, exist_ok=True)
    (outdir / "blend").mkdir(parents=True, exist_ok=True)

    # 1. Merge OOFs
    print("[Meta] Merging OOF files...")
    y, Z, folds, merged, labels = merge_oofs(args.oof_common, args.labels)
    
    print(f"[Meta] Input shape: {Z.shape}. Models: {labels}")
    
    # 2. Base Metrics
    base_scores = {}
    for i, lab in enumerate(labels):
        score = root_mse(y, Z[:, i])
        base_scores[lab] = score
        print(f"  > {lab}: RMSE={score:.4f}")

    # 3. Stacking (Ridge)
    print("[Meta] Running Stacking (Ridge)...")
    stack_preds, stack_model, stack_alpha = stacking_cv(Z, y, folds, seed=args.seed)
    stack_rmse = root_mse(y, stack_preds)
    
    # 4. Blending (Linear)
    print("[Meta] Running Blending (Simplex)...")
    blend_preds, blend_weights_vec = blending_cv(Z, y, folds)
    blend_rmse = root_mse(y, blend_preds)
    
    print(f"[Meta] Stacking RMSE: {stack_rmse:.4f} (alpha={stack_alpha})")
    print(f"[Meta] Blending RMSE: {blend_rmse:.4f}")

    # 5. Save Artifacts
    
    # A. Combined Predictions
    merged["oof_stack"] = stack_preds
    merged["oof_blend"] = blend_preds
    out_parquet = outdir / f"oof_predictions.parquet"
    merged.to_parquet(out_parquet, index=False)
    
    # B. Stacking Model (Pickle for inference)
    # We save metadata to map columns correctly during inference
    feature_map = {
        "xgb": "y_xgb", "lgbm": "y_lgbm", 
        "gnn": "y_chemprop", "chemprop": "y_chemprop", 
        "tpsa": "y_tpsa"
    }
    feature_names = [feature_map.get(lab, f"y_{lab}") for lab in labels]
    
    stack_artifact = {
        "model_type": "ridge",
        "labels": labels,
        "feature_names": feature_names,
        "coef": stack_model.coef_.tolist(),
        "intercept": stack_model.intercept_,
        "alpha": stack_alpha
    }
    with open(outdir / f"stack/meta_ridge.pkl", "wb") as f:
        pickle.dump(stack_artifact, f)
        
    # C. Blending Weights (JSON)
    weights_dict = dict(zip(labels, blend_weights_vec.tolist()))
    with open(outdir / f"blend/weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
        
    # D. Metrics Report
    report = {
        "base_models": base_scores,
        "stacking": {"rmse": stack_rmse, "r2": r2(y, stack_preds), "alpha": stack_alpha},
        "blending": {"rmse": blend_rmse, "r2": r2(y, blend_preds), "weights": weights_dict},
        "winner": "stack" if stack_rmse < blend_rmse else "blend"
    }
    with open(outdir / f"metrics_oof.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"[Meta] Done. Winner: {report['winner']}")

if __name__ == "__main__":
    main()