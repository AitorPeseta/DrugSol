#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_gbm.py
-----------------
Retrains XGBoost and LightGBM on the full dataset using optimized hyperparameters.
- Aggregates best HPs from OOF folds (median/mode).
- Supports GPU acceleration.
- Exports complete Pipelines (Preprocessing + Model) as pickles.
"""

import argparse
import json
import sys
import pickle
import statistics
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

import xgboost as xgb
import lightgbm as lgb

RANDOM_STATE = 42

def read_any(path):
    p = Path(path)
    if p.suffix == '.parquet': return pd.read_parquet(p)
    return pd.read_csv(p)

def get_aggregated_params(hp_dir, prefix):
    """Aggregates params from multiple fold JSONs (median for float, mode for int/str)."""
    p = Path(hp_dir)
    files = sorted(p.glob(f"{prefix}_fold*.json"))
    if not files: return {}
    
    all_params = [json.loads(f.read_text()) for f in files]
    if not all_params: return {}
    
    keys = all_params[0].keys()
    agg = {}
    
    for k in keys:
        vals = [d[k] for d in all_params if k in d]
        if not vals: continue
        
        # Check type of first value
        if isinstance(vals[0], float):
            agg[k] = float(np.median(vals))
        elif isinstance(vals[0], int) and not isinstance(vals[0], bool):
             # Integers (like max_depth) should be rounded median or mode
             agg[k] = int(round(np.median(vals)))
        else:
            # Strings/Bools -> Mode
            try:
                agg[k] = statistics.mode(vals)
            except:
                agg[k] = vals[0] # Fallback
                
    return agg

def get_pipeline_xgb(params, use_gpu):
    # Base
    p = {
        'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 8,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE, 'n_jobs': 4, 'verbosity': 0
    }
    # GPU
    if use_gpu:
        p.update({'tree_method': 'gpu_hist', 'device': 'cuda'})
    else:
        p.update({'tree_method': 'hist', 'device': 'cpu'})
        
    p.update(params)
    
    return xgb.XGBRegressor(**p)

def get_pipeline_lgbm(params, use_gpu):
    p = {
        'n_estimators': 3000, 'learning_rate': 0.03, 'num_leaves': 128,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE, 'n_jobs': 4, 'verbose': -1
    }
    if use_gpu:
        p.update({'device_type': 'gpu'})
    else:
        p.update({'device_type': 'cpu'})
        
    p.update(params)
    return lgb.LGBMRegressor(**p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", default="logS")
    ap.add_argument("--hp-dir", required=True)
    ap.add_argument("--save-dir", default=".")
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--sample-weight-col", default=None)
    args = ap.parse_args()

    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("[Full GBM] Loading data...")
    df = read_any(args.train)
    
    y = df[args.target].values
    
    # Weights
    if args.sample_weight_col and args.sample_weight_col in df.columns:
        w = df[args.sample_weight_col].fillna(1.0).values
    else:
        w = None

    # Drop non-features
    drop_cols = [args.target, args.sample_weight_col, "row_uid", "fold", "smiles_neutral"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=np.number)
    
    # 2. Preprocessing Pipeline
    # It's critical to save the imputer logic
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('var_thresh', VarianceThreshold(threshold=0.0))
    ])

    # 3. Load Hyperparams
    hp_xgb = get_aggregated_params(args.hp_dir, "xgb")
    hp_lgb = get_aggregated_params(args.hp_dir, "lgbm")
    
    print(f"[Full GBM] Best XGB Params: {hp_xgb}")
    print(f"[Full GBM] Best LGB Params: {hp_lgb}")

    # 4. Train XGBoost
    print("[Full GBM] Training XGBoost...")
    xgb_model = get_pipeline_xgb(hp_xgb, args.use_gpu)
    
    xgb_pipe = Pipeline([('pre', preprocessor), ('model', xgb_model)])
    
    # XGBoost supports sample_weight in fit
    fit_params = {}
    if w is not None: fit_params['model__sample_weight'] = w
    
    xgb_pipe.fit(X, y, **fit_params)
    
    with open(outdir / "xgb.pkl", "wb") as f:
        pickle.dump(xgb_pipe, f)

    # 5. Train LightGBM
    print("[Full GBM] Training LightGBM...")
    # Fallback logic for GPU failure
    try:
        lgb_model = get_pipeline_lgbm(hp_lgb, args.use_gpu)
        lgb_pipe = Pipeline([('pre', preprocessor), ('model', lgb_model)])
        
        fit_params_lgb = {}
        if w is not None: fit_params_lgb['model__sample_weight'] = w
        
        lgb_pipe.fit(X, y, **fit_params_lgb)
        
    except Exception as e:
        if args.use_gpu:
            print(f"[WARN] LightGBM GPU failed ({e}). Retrying on CPU...")
            lgb_model = get_pipeline_lgbm(hp_lgb, use_gpu=False)
            lgb_pipe = Pipeline([('pre', preprocessor), ('model', lgb_model)])
            lgb_pipe.fit(X, y, **fit_params_lgb)
        else:
            raise e

    with open(outdir / "lgbm.pkl", "wb") as f:
        pickle.dump(lgb_pipe, f)

    # Save Metadata
    manifest = {
        "features": list(X.columns),
        "n_train": len(df),
        "xgb_params": hp_xgb,
        "lgb_params": hp_lgb
    }
    (outdir / "gbm_manifest.json").write_text(json.dumps(manifest, indent=2))
    
    print(f"[Full GBM] Done. Models saved to {outdir}")

if __name__ == "__main__":
    main()