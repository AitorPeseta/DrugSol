#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Full Physics: Final Physics-Informed Model Training
==========================================================

Trains the final physics-informed Ridge Regression model on the complete
dataset. Uses interpretable physicochemical features based on molecular
properties and thermodynamic principles.

Physics-Based Features:
    - LogP: Octanol-water partition coefficient (hydrophobicity)
    - TPSA: Topological polar surface area (polarity indicator)
    - MW: Molecular weight (molecular size)
    - 1/T: Inverse temperature in Kelvin (thermodynamic term)

The model learns a linear relationship:
    logS = β₀ + β₁·(1000/T_K) + β₂·LogP + β₃·TPSA + β₄·MW

Output Formats:
    PKL Format:
        - Scikit-learn Pipeline (StandardScaler + Ridge)
        - For inference in Python with sklearn
    
    JSON Format:
        - De-standardized raw coefficients
        - Portable inference without Python dependencies
        - Formula: logS = intercept + Σ(coef_i × feature_i)

Arguments:
    --train    : Training data file (Parquet/CSV)
    --target   : Target column name (default: logS)
    --save-dir : Output directory
    --seed     : Random seed

Usage:
    python train_full_physics.py \\
        --train train_data.parquet \\
        --target logS \\
        --save-dir models_physics

Output:
    - physics_ridge.pkl: Sklearn Pipeline for Python inference
    - physics_model.json: Raw coefficients for portable inference

Notes:
    - Missing temperature defaults to 25°C (room temperature)
    - Feature scaling is applied internally
    - JSON coefficients are de-standardized for direct use
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline


# ============================================================================
# UTILITIES
# ============================================================================

def read_any(path: str) -> pd.DataFrame:
    """Read Parquet or CSV file."""
    p = Path(path)
    if p.suffix == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)


def get_column_name(df: pd.DataFrame, candidates: list) -> str:
    """Find the first matching column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for full physics model training."""
    
    ap = argparse.ArgumentParser(
        description="Train full physics-informed Ridge Regression model."
    )
    ap.add_argument("--train", required=True,
                    help="Training data file")
    ap.add_argument("--target", default="logS",
                    help="Target column")
    ap.add_argument("--save-dir", default=".",
                    help="Output directory")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    
    args = ap.parse_args()
    
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"[Full Physics] Loading {args.train}...")
    df = read_any(args.train)
    print(f"[Full Physics] Loaded {len(df):,} samples")
    
    # -------------------------------------------------------------------------
    # Resolve Feature Columns
    # -------------------------------------------------------------------------
    col_temp = get_column_name(df, ["temp_C", "temperature", "Temperature"])
    col_logp = get_column_name(df, ["rdkit__logP", "logP", "MolLogP"])
    col_tpsa = get_column_name(df, ["rdkit__TPSA", "TPSA", "tpsa"])
    col_mw = get_column_name(df, ["rdkit__MW", "MW", "MolWt"])
    
    features = []
    
    # Temperature → 1/T (thermodynamic term)
    if col_temp:
        print(f"[Full Physics] Temperature column: {col_temp}")
        # Fill missing with 25°C, convert to Kelvin
        t_celsius = df[col_temp].fillna(25.0)
        t_kelvin = t_celsius + 273.15
        df["inv_temp"] = 1000.0 / t_kelvin
        features.append("inv_temp")
    else:
        print("[Full Physics] No temperature column. Assuming 25°C.")
        df["inv_temp"] = 1000.0 / 298.15
        features.append("inv_temp")
    
    # Molecular properties
    feature_map = {"LogP": col_logp, "TPSA": col_tpsa, "MW": col_mw}
    
    for name, col in feature_map.items():
        if col:
            print(f"[Full Physics] {name} column: {col}")
            features.append(col)
            # Impute missing with median
            df[col] = df[col].fillna(df[col].median())
        else:
            sys.exit(f"[ERROR] Required column '{name}' not found in dataset")
    
    print(f"\n[Full Physics] Features: {features}")
    
    # -------------------------------------------------------------------------
    # Prepare Data
    # -------------------------------------------------------------------------
    df_clean = df.dropna(subset=[args.target] + features)
    X = df_clean[features].values
    y = df_clean[args.target].values
    
    if len(X) == 0:
        sys.exit("[ERROR] Dataset empty after dropping NaNs")
    
    print(f"[Full Physics] Training samples: {len(X):,}")
    
    # -------------------------------------------------------------------------
    # Train Model
    # -------------------------------------------------------------------------
    # Pipeline: StandardScaler → RidgeCV
    # Scaling is important because features have different magnitudes
    # (MW ~300, inv_temp ~3.3, TPSA ~80)
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]))
    ])
    
    print("[Full Physics] Training Ridge Regression...")
    pipe.fit(X, y)
    
    # -------------------------------------------------------------------------
    # Save Pickle
    # -------------------------------------------------------------------------
    joblib.dump(pipe, outdir / "physics_ridge.pkl")
    print(f"[Full Physics] Saved pipeline to physics_ridge.pkl")
    
    # -------------------------------------------------------------------------
    # Export Raw Coefficients (De-standardization)
    # -------------------------------------------------------------------------
    # The sklearn pipeline applies: y = ridge(scaler(X))
    # Where scaler(X) = (X - mean) / scale
    # So: y = w @ ((X - mean) / scale) + b
    #       = (w / scale) @ X - (w @ mean / scale) + b
    # 
    # Raw coefficients: w_raw = w / scale
    # Raw intercept: b_raw = b - sum(w * mean / scale)
    
    scaler = pipe.named_steps["scaler"]
    ridge = pipe.named_steps["ridge"]
    
    w_scaled = ridge.coef_
    b_scaled = ridge.intercept_
    means = scaler.mean_
    scales = scaler.scale_
    
    # De-standardize
    w_raw = w_scaled / scales
    b_raw = b_scaled - np.sum(w_scaled * means / scales)
    
    # Build coefficient dictionary
    coef_dict = {f: float(c) for f, c in zip(features, w_raw)}
    
    model_json = {
        "model_type": "ridge_physics_informed",
        "intercept": float(b_raw),
        "coefficients": coef_dict,
        "features": features,
        "feature_order": features,
        "alpha": float(ridge.alpha_),
        "n_train": len(X),
        "formula": f"logS = {b_raw:.4f} + " + " + ".join([f"{c:.4f}*{f}" for f, c in coef_dict.items()])
    }
    
    with open(outdir / "physics_model.json", "w") as f:
        json.dump(model_json, f, indent=2)
    
    print(f"[Full Physics] Saved coefficients to physics_model.json")
    
    # -------------------------------------------------------------------------
    # Report Results
    # -------------------------------------------------------------------------
    print(f"\n[Full Physics] Model Summary:")
    print(f"         Intercept: {b_raw:.4f}")
    for feat, coef in coef_dict.items():
        print(f"         {feat}: {coef:.4f}")
    print(f"         Best alpha: {ridge.alpha_}")
    print(f"\n[Full Physics] Formula:")
    print(f"         {model_json['formula']}")


if __name__ == "__main__":
    main()
