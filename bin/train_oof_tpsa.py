#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_oof_tpsa.py
-----------------
Trains a Physics-Informed Ridge Regression Baseline.
Features: 
  - LogP, TPSA, MW (Proxies for polarity and size).
  - 1/T (Van 't Hoff dependency).
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--folds-file", required=True)
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="oof_tpsa")
    args = ap.parse_args()

    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("[Baseline] Loading data...")
    df = read_any(args.train)
    folds = read_any(args.folds_file)
    
    # Merge folds
    df = df.merge(folds[[args.id_col, "fold"]], on=args.id_col, how="inner")

    # 2. Feature Engineering (Physics)
    # Identify columns (flexible naming)
    col_temp = get_column_name(df, ["temp_C", "temperature"])
    col_logp = get_column_name(df, ["rdkit__logP", "logP", "MolLogP"])
    col_tpsa = get_column_name(df, ["rdkit__TPSA", "TPSA"])
    col_mw   = get_column_name(df, ["rdkit__MW", "MW", "MolWt"])
    
    features = []
    
    # Temperature (1/T)
    if col_temp:
        print(f"  > Using Temperature: {col_temp}")
        # Fill NaNs with RTP (25 C)
        t_k = df[col_temp].fillna(25.0) + 273.15
        df["inv_temp"] = 1000.0 / t_k
        features.append("inv_temp")
    else:
        print("  [WARN] No temperature column found!")

    # Physicochemical Props
    for name, col in [("LogP", col_logp), ("TPSA", col_tpsa), ("MW", col_mw)]:
        if col:
            print(f"  > Using {name}: {col}")
            features.append(col)
            # Fill NaNs with median
            df[col] = df[col].fillna(df[col].median())
        else:
            print(f"  [WARN] {name} column not found.")

    if not features:
        sys.exit("[ERROR] No features found for Baseline model.")

    print(f"[Baseline] Training Ridge on: {features}")

    # 3. Train OOF
    y = df[args.target].values
    X = df[features].values
    folds_vec = df["fold"].values
    oof_preds = np.zeros(len(y))
    
    unique_folds = sorted(np.unique(folds_vec))
    
    # RidgeCV automatically finds best alpha
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
    ])

    for k in unique_folds:
        val_mask = (folds_vec == k)
        train_mask = ~val_mask
        
        model.fit(X[train_mask], y[train_mask])
        oof_preds[val_mask] = model.predict(X[val_mask])

    # 4. Metrics & Save
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    r2 = r2_score(y, oof_preds)
    
    print(f"[Baseline] OOF RMSE: {rmse:.4f} | R2: {r2:.4f}")
    
    # Save OOF
    out_df = pd.DataFrame({
        "id": df[args.id_col].astype(str),
        "fold": df["fold"].astype(int),
        "y_true": y,
        "y_pred": oof_preds,
        "model": "ridge_baseline"
    })
    out_df.to_parquet(outdir / "oof_tpsa.parquet", index=False)
    
    # Save metrics
    metrics = {"rmse": float(rmse), "r2": float(r2)}
    (outdir / "metrics_oof_tpsa.json").write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()