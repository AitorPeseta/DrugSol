#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_tpsa.py
------------------
Trains the final 'Physically Inspired Ridge' model on the full dataset.
Features: LogP, TPSA, MW, 1/Temperature.

Outputs:
1. .pkl: Scikit-learn Pipeline (StandardScaler + Ridge).
2. .json: Raw weights (de-standardized) for portable inference without Python/sklearn.
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

def read_any(path):
    p = Path(path)
    if p.suffix == '.parquet': return pd.read_parquet(p)
    return pd.read_csv(p)

def get_column_name(df, candidates):
    """Finds the first matching column from candidates."""
    for c in candidates:
        if c in df.columns: return c
    return None

def main():
    ap = argparse.ArgumentParser(description="Train Full Ridge Baseline.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default=".")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print(f"[Full TPSA] Loading {args.train}...")
    df = read_any(args.train)

    # 2. Resolve Feature Columns
    # We look for rdkit__ prefix (from make_features_rdkit) or raw names
    col_temp = get_column_name(df, ["temp_C", "temperature"])
    col_logp = get_column_name(df, ["rdkit__logP", "logP", "MolLogP"])
    col_tpsa = get_column_name(df, ["rdkit__TPSA", "TPSA"])
    col_mw   = get_column_name(df, ["rdkit__MW", "MW", "MolWt"])

    features = []
    
    # Temperature Engineering (1/T Kelvin)
    if col_temp:
        # Fill missing temps with 25 C (RTP)
        t_k = df[col_temp].fillna(25.0) + 273.15
        df["inv_temp"] = 1000.0 / t_k
        features.append("inv_temp")
    else:
        print("[WARN] No temperature column. Assuming 25C.")
        df["inv_temp"] = 1000.0 / 298.15
        features.append("inv_temp")

    # Molecular Props
    feature_map = {"logP": col_logp, "TPSA": col_tpsa, "MW": col_mw}
    
    for name, col in feature_map.items():
        if col:
            features.append(col)
            # Simple imputation for safety
            df[col] = df[col].fillna(df[col].median())
        else:
            sys.exit(f"[ERROR] Critical column '{name}' not found in dataset.")

    print(f"[Full TPSA] Features: {features}")

    # 3. Prepare Arrays
    df_clean = df.dropna(subset=[args.target] + features)
    X = df_clean[features].values
    y = df_clean[args.target].values

    if len(X) == 0:
        sys.exit("[ERROR] Dataset empty after dropping NaNs.")

    # 4. Train (Pipeline: Scale -> RidgeCV)
    # Scaling is crucial because MW (~300) and inv_temp (~3.3) have different magnitudes
    pipe = Pipeline([
        ("scaler", StandardScaler()), 
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
    ])
    
    print(f"[Full TPSA] Training on {len(X)} samples...")
    pipe.fit(X, y)

    # 5. Save Pickle
    joblib.dump(pipe, outdir / "tpsa_phys.pkl")

    # 6. Export Raw Mathematics (De-standardization)
    # Formula: y = b_raw + w1_raw*x1 + ...
    # w_raw = w_scaled / sigma
    # b_raw = b_scaled - sum(w_scaled * mu / sigma)
    
    scaler = pipe.named_steps["scaler"]
    ridge = pipe.named_steps["ridge"]
    
    w = ridge.coef_
    b = ridge.intercept_
    
    means = scaler.mean_
    scales = scaler.scale_
    
    raw_coefs = w / scales
    raw_intercept = b - np.sum(w * means / scales)
    
    # Save as JSON
    coef_dict = {f: float(c) for f, c in zip(features, raw_coefs)}
    
    model_json = {
        "model_type": "Ridge_Physics_Based",
        "intercept": float(raw_intercept),
        "coefs": coef_dict,
        "features": features,
        "alpha": float(ridge.alpha_),
        "training_samples": len(X)
    }
    
    (outdir / "tpsa_model.json").write_text(json.dumps(model_json, indent=2))
    print(f"[Full TPSA] Done. Intercept: {raw_intercept:.4f}")

if __name__ == "__main__":
    main()