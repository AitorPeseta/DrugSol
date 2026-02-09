#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train OOF Physics: Physics-Informed Baseline Model
===================================================

Trains a physics-informed Ridge Regression baseline using interpretable
molecular features and thermodynamic principles. Provides complementary
predictions for ensemble stacking.

Physics-Based Features:
    - LogP: Octanol-water partition coefficient (hydrophobicity)
    - TPSA: Topological polar surface area (polarity indicator)
    - MW: Molecular weight (size/complexity)
    - 1/T: Inverse temperature in Kelvin (Van't Hoff thermodynamic term)

Van't Hoff Equation:
    The temperature dependency of solubility follows:
    
        ln(S) = -ΔH°/R × (1/T) + ΔS°/R
    
    By including 1/T as a feature, the model captures the enthalpy-driven
    component of solubility. The coefficient learned for 1/T is proportional
    to -ΔH°/R, providing physical interpretability.

Final Equation:
    logS = β₀ + β₁·(1/T) + β₂·LogP + β₃·TPSA + β₄·MW

Model Selection:
    Ridge Regression was chosen for:
    - Stability with correlated features
    - Automatic regularization (RidgeCV)
    - Fast training
    - Interpretable coefficients

Arguments:
    --train      : Training data file (Parquet/CSV)
    --folds-file : Fold assignments file (Parquet/CSV)
    --id-col     : Row identifier column (default: row_uid)
    --target     : Target column name (default: logS)
    --save-dir   : Output directory (default: oof_physics)

Usage:
    python train_oof_physics.py \\
        --train train_data.parquet \\
        --folds-file folds.parquet \\
        --id-col row_uid \\
        --target logS \\
        --save-dir oof_physics

Output:
    - oof_physics.parquet: OOF predictions with columns [id, fold, y_true, y_pred, model]
    - metrics_oof_physics.json: Performance metrics (RMSE, R²)

Notes:
    - Missing temperature is filled with 25°C (room temperature assumption)
    - Missing features are filled with column median
    - Feature standardization is applied before Ridge regression
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================================
# I/O UTILITIES
# ============================================================================

def read_any(path: str) -> pd.DataFrame:
    """Read Parquet or CSV file."""
    p = Path(path)
    if p.suffix == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)


def get_column_name(df: pd.DataFrame, candidates: list) -> str:
    """
    Find the first matching column from a list of candidates.
    
    Args:
        df: DataFrame to search
        candidates: List of possible column names
    
    Returns:
        First matching column name, or None if no match
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for physics baseline training."""
    
    ap = argparse.ArgumentParser(
        description="Train physics-informed Ridge Regression baseline."
    )
    ap.add_argument("--train", required=True,
                    help="Training data file")
    ap.add_argument("--folds-file", required=True,
                    help="Fold assignments file")
    ap.add_argument("--id-col", default="row_uid",
                    help="ID column (default: row_uid)")
    ap.add_argument("--target", default="logS",
                    help="Target column (default: logS)")
    ap.add_argument("--save-dir", default="oof_physics",
                    help="Output directory")
    
    args = ap.parse_args()
    
    # Setup output directory
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("[Physics] Loading data...")
    df = read_any(args.train)
    folds = read_any(args.folds_file)
    
    # Merge folds
    df = df.merge(folds[[args.id_col, "fold"]], on=args.id_col, how="inner")
    print(f"[Physics] Loaded {len(df):,} samples")
    
    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    print("[Physics] Identifying features...")
    
    # Find columns with flexible naming
    col_temp = get_column_name(df, ["temp_C", "temperature", "Temperature"])
    col_logp = get_column_name(df, ["rdkit__logP", "logP", "MolLogP", "LogP"])
    col_tpsa = get_column_name(df, ["rdkit__TPSA", "TPSA", "tpsa"])
    col_mw = get_column_name(df, ["rdkit__MW", "MW", "MolWt", "mw"])
    
    features = []
    
    # Temperature → 1/T (Van't Hoff term)
    if col_temp:
        print(f"         Temperature: {col_temp}")
        # Fill NaN with room temperature (25°C), convert to Kelvin
        t_celsius = df[col_temp].fillna(25.0)
        t_kelvin = t_celsius + 273.15
        # Use 1000/T for better numerical scaling
        df["inv_temp"] = 1000.0 / t_kelvin
        features.append("inv_temp")
    else:
        print("         [WARN] No temperature column found")
    
    # Physicochemical properties
    for name, col in [("LogP", col_logp), ("TPSA", col_tpsa), ("MW", col_mw)]:
        if col:
            print(f"         {name}: {col}")
            features.append(col)
            # Fill NaN with median
            df[col] = df[col].fillna(df[col].median())
        else:
            print(f"         [WARN] {name} column not found")
    
    if not features:
        sys.exit("[ERROR] No features available for Physics baseline model")
    
    print(f"\n[Physics] Training Ridge Regression on {len(features)} features:")
    print(f"         {features}")
    
    # -------------------------------------------------------------------------
    # Prepare Data
    # -------------------------------------------------------------------------
    y = df[args.target].values
    X = df[features].values
    folds_vec = df["fold"].values
    oof_preds = np.zeros(len(y))
    
    unique_folds = sorted(np.unique(folds_vec))
    
    # -------------------------------------------------------------------------
    # Train with Out-of-Fold Predictions
    # -------------------------------------------------------------------------
    # RidgeCV automatically selects best regularization strength
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]))
    ])
    
    print(f"\n[Physics] Training across {len(unique_folds)} folds...")
    
    for k in unique_folds:
        val_mask = (folds_vec == k)
        train_mask = ~val_mask
        
        model.fit(X[train_mask], y[train_mask])
        oof_preds[val_mask] = model.predict(X[val_mask])
        
        # Report fold metrics
        fold_rmse = np.sqrt(mean_squared_error(y[val_mask], oof_preds[val_mask]))
        print(f"         Fold {k}: RMSE = {fold_rmse:.4f}")
    
    # -------------------------------------------------------------------------
    # Calculate Overall Metrics
    # -------------------------------------------------------------------------
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    r2 = r2_score(y, oof_preds)
    
    print(f"\n[Physics] OOF Results:")
    print(f"         RMSE: {rmse:.4f}")
    print(f"         R²: {r2:.4f}")
    
    # Report learned coefficients from last fold (for interpretability)
    ridge_model = model.named_steps['ridge']
    scaler = model.named_steps['scaler']
    
    print(f"\n[Physics] Feature Coefficients (standardized):")
    for feat, coef in zip(features, ridge_model.coef_):
        print(f"         {feat}: {coef:.4f}")
    print(f"         Intercept: {ridge_model.intercept_:.4f}")
    print(f"         Best alpha: {ridge_model.alpha_:.4f}")
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    # OOF predictions
    out_df = pd.DataFrame({
        "id": df[args.id_col].astype(str),
        "fold": df["fold"].astype(int),
        "y_true": y,
        "y_pred": oof_preds,
        "model": "physics_baseline"
    })
    out_df.to_parquet(outdir / "oof_physics.parquet", index=False)
    
    # Metrics
    metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "features": features,
        "best_alpha": float(ridge_model.alpha_)
    }
    with open(outdir / "metrics_oof_physics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Physics] Saved to {outdir}/")


if __name__ == "__main__":
    main()
