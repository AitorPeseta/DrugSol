#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta Stack Blend: Ensemble Meta-Learning Module
================================================

Combines Out-of-Fold (OOF) predictions from multiple base models using
two ensemble strategies: Stacking (Ridge Regression) and Blending
(Simplex-constrained linear combination).

Ensemble Strategies:
    Stacking:
        - Uses Ridge Regression to learn optimal combination weights
        - Allows negative weights and weights > 1
        - Cross-validated alpha selection
        - Generally more flexible
    
    Blending:
        - Constrains weights to be non-negative and sum to 1
        - Uses simplex projection for constraint satisfaction
        - More interpretable weights
        - Often more robust to overfitting

Arguments:
    --oof-common : List of OOF prediction parquet files
    --labels     : Model labels corresponding to each file
    --metric     : Selection metric (rmse or r2)
    --save-dir   : Output directory
    --seed       : Random seed

Usage:
    python meta_stack_blend.py \\
        --oof-common xgb.parquet lgbm.parquet cat.parquet gnn.parquet physics.parquet \\
        --labels xgb lgbm cat gnn physics \\
        --metric rmse \\
        --save-dir meta_results

Output:
    - stack/meta_ridge.pkl: Stacking meta-model
    - blend/weights.json: Blending weights
    - oof_predictions.parquet: Combined OOF predictions
    - metrics_oof.json: Performance comparison

Notes:
    - OOF files must contain columns: id, fold, y_true, y_pred
    - Duplicate (id, fold) pairs are aggregated by mean
    - Winner is selected based on cross-validated RMSE
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================================
# METRICS
# ============================================================================

def root_mse(y, yhat):
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y, yhat)))


def r2(y, yhat):
    """Calculate R-squared score."""
    return float(r2_score(y, yhat))


# ============================================================================
# DATA UTILITIES
# ============================================================================

def read_any(path: str) -> pd.DataFrame:
    """Read Parquet or CSV file."""
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path}")


def collapse_oof(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate (id, fold) pairs by averaging predictions."""
    return df.groupby(["id", "fold"], as_index=False).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean")
    )


def merge_oofs(oof_files, labels):
    """
    Merge multiple OOF prediction files into a single DataFrame.
    
    Args:
        oof_files: List of file paths
        labels: List of model labels
    
    Returns:
        Tuple of (y, Z, folds, merged_df, final_labels)
    """
    merged = None
    final_labels = []
    
    for i, f in enumerate(oof_files):
        df = read_any(f)
        label = labels[i] if labels and i < len(labels) else f"model_{i}"
        final_labels.append(label)
        
        # Validate required columns
        req = {"id", "fold", "y_true", "y_pred"}
        if not req.issubset(df.columns):
            raise ValueError(f"File {f} missing columns. Required: {req}")
        
        # Handle duplicates
        if df.duplicated(subset=["id", "fold"]).any():
            df = collapse_oof(df)
        
        # Rename columns for this model
        df_renamed = df[["id", "fold", "y_true", "y_pred"]].rename(
            columns={"y_true": f"y_true_{label}", "y_pred": f"oof_{label}"}
        )
        
        if merged is None:
            merged = df_renamed
        else:
            merged = merged.merge(df_renamed, on=["id", "fold"], how="inner")
    
    if merged is None or merged.empty:
        raise ValueError("Merge resulted in empty dataframe")
    
    # Extract arrays
    y_col = f"y_true_{final_labels[0]}"
    y = merged[y_col].values
    Z = merged[[f"oof_{lab}" for lab in final_labels]].values
    folds = merged["fold"].values
    
    return y, Z, folds, merged, final_labels


# ============================================================================
# BLENDING (Simplex Projection)
# ============================================================================

def simplex_project(v):
    """
    Project vector onto probability simplex (non-negative, sum to 1).
    
    Uses the algorithm from "Efficient Projections onto the L1-Ball
    for Learning in High Dimensions" (Duchi et al., 2008).
    """
    v = np.array(v, dtype=float)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(n) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def blend_weights(Z, y, l2=1e-8):
    """
    Compute blending weights with simplex constraint.
    
    Solves: min ||Zw - y||^2 + l2||w||^2
    Subject to: w >= 0, sum(w) = 1
    """
    Z = np.array(Z, dtype=float)
    y = np.array(y, dtype=float)
    n_models = Z.shape[1]
    
    # Regularized least squares
    A = Z.T @ Z + l2 * np.eye(n_models)
    b = Z.T @ y
    
    try:
        w_unconstrained = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_unconstrained = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Project onto simplex
    return simplex_project(w_unconstrained)


def blending_cv(Z, y, folds):
    """
    Cross-validated blending with simplex-constrained weights.
    
    Returns:
        Tuple of (oof_predictions, final_weights)
    """
    yhat = np.zeros_like(y)
    unique_folds = np.unique(folds)
    
    for k in unique_folds:
        val_idx = (folds == k)
        trn_idx = ~val_idx
        w_k = blend_weights(Z[trn_idx], y[trn_idx])
        yhat[val_idx] = Z[val_idx] @ w_k
    
    # Final weights on full data
    w_full = blend_weights(Z, y)
    return yhat, w_full


# ============================================================================
# STACKING (Ridge Regression)
# ============================================================================

def stacking_cv(Z, y, folds, seed=42):
    """
    Cross-validated stacking with Ridge Regression.
    
    Returns:
        Tuple of (oof_predictions, fitted_model, best_alpha)
    """
    yhat = np.zeros_like(y)
    unique_folds = np.unique(folds)
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Find best alpha via internal CV
    ridge_cv = RidgeCV(alphas=alphas, cv=KFold(5, shuffle=True, random_state=seed))
    ridge_cv.fit(Z, y)
    best_alpha = ridge_cv.alpha_
    
    # Generate OOF predictions
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


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for meta stack blend."""
    
    ap = argparse.ArgumentParser(
        description="Ensemble meta-learning via stacking and blending."
    )
    ap.add_argument("--oof-common", nargs="+", required=True,
                    help="List of OOF parquet files")
    ap.add_argument("--labels", nargs="+",
                    help="Model labels (e.g., xgb lgbm cat gnn physics)")
    ap.add_argument("--metric", choices=["rmse", "r2"], default="rmse",
                    help="Selection metric")
    ap.add_argument("--save-dir", default="meta_results",
                    help="Output directory")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    
    args = ap.parse_args()
    
    # Setup output directories
    outdir = Path(args.save_dir)
    (outdir / "stack").mkdir(parents=True, exist_ok=True)
    (outdir / "blend").mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Merge OOF Predictions
    # -------------------------------------------------------------------------
    print("[Meta] Merging OOF files...")
    y, Z, folds, merged, labels = merge_oofs(args.oof_common, args.labels)
    print(f"[Meta] Input shape: {Z.shape}, Models: {labels}")
    
    # Report base model scores
    print("\n[Meta] Base Model Performance:")
    base_scores = {}
    for i, lab in enumerate(labels):
        score = root_mse(y, Z[:, i])
        base_scores[lab] = score
        print(f"       {lab}: RMSE = {score:.4f}")
    
    # -------------------------------------------------------------------------
    # Stacking
    # -------------------------------------------------------------------------
    print("\n[Meta] Running Stacking (Ridge Regression)...")
    stack_preds, stack_model, stack_alpha = stacking_cv(Z, y, folds, seed=args.seed)
    stack_rmse = root_mse(y, stack_preds)
    stack_r2 = r2(y, stack_preds)
    
    print(f"       RMSE: {stack_rmse:.4f}")
    print(f"       R²: {stack_r2:.4f}")
    print(f"       Alpha: {stack_alpha}")
    print(f"       Coefficients: {dict(zip(labels, stack_model.coef_.round(4)))}")
    
    # -------------------------------------------------------------------------
    # Blending
    # -------------------------------------------------------------------------
    print("\n[Meta] Running Blending (Simplex Projection)...")
    blend_preds, blend_weights_vec = blending_cv(Z, y, folds)
    blend_rmse = root_mse(y, blend_preds)
    blend_r2 = r2(y, blend_preds)
    
    weights_dict = dict(zip(labels, blend_weights_vec.tolist()))
    print(f"       RMSE: {blend_rmse:.4f}")
    print(f"       R²: {blend_r2:.4f}")
    print(f"       Weights: {dict(zip(labels, blend_weights_vec.round(4)))}")
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    # Combined predictions
    merged["oof_stack"] = stack_preds
    merged["oof_blend"] = blend_preds
    merged.to_parquet(outdir / "oof_predictions.parquet", index=False)
    
    # Stacking model
    feature_map = {
        "xgb": "y_xgb", "lgbm": "y_lgbm", "cat": "y_cat",
        "gnn": "y_chemprop", "chemprop": "y_chemprop",
        "physics": "y_physics"
    }
    feature_names = [feature_map.get(lab, f"y_{lab}") for lab in labels]
    
    stack_artifact = {
        "model_type": "ridge",
        "labels": labels,
        "feature_names": feature_names,
        "coef": stack_model.coef_.tolist(),
        "intercept": float(stack_model.intercept_),
        "alpha": float(stack_alpha)
    }
    with open(outdir / "stack/meta_ridge.pkl", "wb") as f:
        pickle.dump(stack_artifact, f)
    
    # Blending weights
    with open(outdir / "blend/weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    
    # Metrics report
    winner = "stack" if stack_rmse < blend_rmse else "blend"
    report = {
        "base_models": base_scores,
        "stacking": {
            "rmse": stack_rmse,
            "r2": stack_r2,
            "alpha": float(stack_alpha),
            "coefficients": dict(zip(labels, stack_model.coef_.tolist()))
        },
        "blending": {
            "rmse": blend_rmse,
            "r2": blend_r2,
            "weights": weights_dict
        },
        "winner": winner
    }
    with open(outdir / "metrics_oof.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[Meta] Winner: {winner.upper()}")
    print(f"[Meta] Saved to {outdir}/")


if __name__ == "__main__":
    main()
