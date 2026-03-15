#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Full GBM: Final Model Training on Complete Dataset
=========================================================

Retrains XGBoost, LightGBM, and CatBoost on the full dataset using
hyperparameters optimized during OOF cross-validation. These models
are used for final inference.

Training Strategy:
    - Aggregates hyperparameters from K-fold tuning (median for numerics)
    - Trains on 100% of data (no validation holdout)
    - Applies sample weighting for temperature importance
    - GPU acceleration with automatic CPU fallback
    - Generates SHAP analysis for XGBoost interpretability

Arguments:
    --train            : Training data file (Parquet/CSV)
    --target           : Target column name (default: logS)
    --hp-dir           : Hyperparameter directory or JSON file
    --save-dir         : Output directory
    --gpu          : Enable GPU acceleration
    --sample-weight-col: Sample weight column

Usage:
    python train_full_gbm.py \\
        --train final_train_gbm.parquet \\
        --target logS \\
        --hp-dir oof_gbm/hp \\
        --gpu \\
        --sample-weight-col weight \\
        --save-dir models_GBM

Output:
    - xgb.pkl: XGBoost model pipeline
    - lgbm.pkl: LightGBM model pipeline
    - cat.pkl: CatBoost model pipeline
    - shap_summary_xgb.png: SHAP feature importance plot
    - gbm_manifest.json: Model metadata

Notes:
    - Models are saved as sklearn Pipelines with preprocessing
    - GPU memory is checked before CatBoost training
    - LightGBM/CatBoost automatically fall back to CPU if GPU fails
"""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Optional SHAP
try:
    import shap
    import matplotlib.pyplot as plt
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

RANDOM_STATE = 42


# ============================================================================
# UTILITIES
# ============================================================================

def read_any(path: str) -> pd.DataFrame:
    """Read Parquet or CSV file."""
    p = Path(path)
    if p.suffix == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)


def check_gpu_memory(min_memory_mb=4000):
    """Check if GPU has enough free memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            free_memory = int(result.stdout.strip().split('\n')[0])
            if free_memory < min_memory_mb:
                print(f"[WARN] GPU memory low: {free_memory}MB free, need {min_memory_mb}MB")
                return False
            return True
    except Exception:
        pass
    return True


def get_aggregated_params(hp_dir, prefix):
    """
    Aggregate hyperparameters from multiple fold JSON files.
    
    For numeric parameters, uses median across folds.
    For categorical parameters, uses first value.
    """
    p = Path(hp_dir)
    
    # If it's a single file, read directly
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    
    # Find fold-specific files
    files = sorted(p.glob(f"{prefix}_fold*.json"))
    if not files:
        return {}
    
    all_params = [json.loads(f.read_text()) for f in files]
    if not all_params:
        return {}
    
    # Aggregate parameters
    keys = all_params[0].keys()
    agg = {}
    
    for k in keys:
        vals = [d[k] for d in all_params if k in d]
        if not vals:
            continue
        
        if isinstance(vals[0], float):
            agg[k] = float(np.median(vals))
        elif isinstance(vals[0], int) and not isinstance(vals[0], bool):
            agg[k] = int(round(np.median(vals)))
        else:
            agg[k] = vals[0]
    
    return agg


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def get_xgb_model(params, use_gpu):
    """Create XGBoost regressor with given parameters."""
    p = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': 4,
        'verbosity': 0
    }
    
    if use_gpu:
        p.update({'tree_method': 'hist', 'device': 'cuda'})
    else:
        p.update({'tree_method': 'hist', 'device': 'cpu'})
    
    p.update(params)
    return xgb.XGBRegressor(**p)


def get_lgbm_model(params, use_gpu):
    """Create LightGBM regressor with given parameters."""
    p = {
        'n_estimators': 3000,
        'learning_rate': 0.03,
        'num_leaves': 128,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': 4,
        'verbose': -1
    }
    
    if use_gpu:
        p.update({'device': 'gpu'})
    else:
        p.update({'device': 'cpu'})
    
    p.update(params)
    return lgb.LGBMRegressor(**p)


def get_cat_model(params, use_gpu):
    """Create CatBoost regressor with given parameters."""
    # Check GPU memory before attempting GPU training
    actual_use_gpu = use_gpu and check_gpu_memory(min_memory_mb=4000)
    
    if use_gpu and not actual_use_gpu:
        print("[WARN] CatBoost: Insufficient GPU memory, using CPU")
    
    p = {
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 8,
        'loss_function': 'RMSE',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'allow_writing_files': False
    }
    
    if actual_use_gpu:
        p.update({'task_type': 'GPU', 'devices': '0'})
    else:
        p.update({'task_type': 'CPU', 'thread_count': 4})
    
    p.update(params)
    return cb.CatBoostRegressor(**p)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for full GBM training."""
    
    ap = argparse.ArgumentParser(
        description="Train full GBM models (XGBoost, LightGBM, CatBoost)."
    )
    ap.add_argument("--train", required=True,
                    help="Training data file")
    ap.add_argument("--target", default="logS",
                    help="Target column")
    ap.add_argument("--hp-dir", required=True,
                    help="Hyperparameter directory or file")
    ap.add_argument("--save-dir", default=".",
                    help="Output directory")
    ap.add_argument("--gpu", dest="use_gpu", action="store_true",
                    help="Enable GPU acceleration")
    ap.add_argument("--sample-weight-col", default=None,
                    help="Sample weight column")
    
    args = ap.parse_args()
    
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("[Full GBM] Loading data...")
    df = read_any(args.train)
    y = df[args.target].values
    
    # Sample weights
    if args.sample_weight_col and args.sample_weight_col in df.columns:
        w = df[args.sample_weight_col].fillna(1.0).values
        print(f"[Full GBM] Using sample weights from '{args.sample_weight_col}'")
    else:
        w = None
    
    # Prepare features
    drop_cols = [args.target, args.sample_weight_col, "row_uid", "fold", "smiles_neutral"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=np.number)
    
    print(f"[Full GBM] Data shape: {X.shape}")
    
    # -------------------------------------------------------------------------
    # Preprocessing Pipeline
    # -------------------------------------------------------------------------
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('var_thresh', VarianceThreshold(threshold=0.0))
    ])
    
    # -------------------------------------------------------------------------
    # Load Hyperparameters
    # -------------------------------------------------------------------------
    hp_xgb = get_aggregated_params(args.hp_dir, "xgb")
    hp_lgb = get_aggregated_params(args.hp_dir, "lgbm")
    hp_cat = get_aggregated_params(args.hp_dir, "cat")
    
    print(f"[Full GBM] XGBoost params: {hp_xgb}")
    print(f"[Full GBM] LightGBM params: {hp_lgb}")
    print(f"[Full GBM] CatBoost params: {hp_cat}")
    
    # -------------------------------------------------------------------------
    # Train XGBoost
    # -------------------------------------------------------------------------
    print("\n[Full GBM] Training XGBoost...")
    xgb_model = get_xgb_model(hp_xgb, args.use_gpu)
    xgb_pipe = Pipeline([('pre', preprocessor), ('model', xgb_model)])
    
    fit_params = {}
    if w is not None:
        fit_params['model__sample_weight'] = w
    
    xgb_pipe.fit(X, y, **fit_params)
    
    with open(outdir / "xgb.pkl", "wb") as f:
        pickle.dump(xgb_pipe, f)
    
    # SHAP Analysis for XGBoost
    if HAS_SHAP:
        try:
            print("[Full GBM] Generating SHAP analysis...")
            inner_model = xgb_pipe.named_steps['model']
            inner_pre = xgb_pipe.named_steps['pre']
            
            X_transformed = inner_pre.transform(X)
            support_mask = inner_pre.named_steps['var_thresh'].get_support()
            final_feats = X.columns[support_mask]
            X_shap = pd.DataFrame(X_transformed, columns=final_feats)
            
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X_shap)
            
            plt.figure(figsize=(10, 12))
            shap.summary_plot(shap_values, X_shap, max_display=20, show=False, plot_type="dot")
            plt.savefig(outdir / "shap_summary_xgb.png", bbox_inches='tight', dpi=300)
            plt.close()
            print("[Full GBM] SHAP plot saved")
        except Exception as e:
            print(f"[WARN] SHAP analysis failed: {e}")
    
    # -------------------------------------------------------------------------
    # Train LightGBM
    # -------------------------------------------------------------------------
    print("\n[Full GBM] Training LightGBM...")
    try:
        lgb_model = get_lgbm_model(hp_lgb, args.use_gpu)
        lgb_pipe = Pipeline([('pre', preprocessor), ('model', lgb_model)])
        
        fit_params_lgb = {}
        if w is not None:
            fit_params_lgb['model__sample_weight'] = w
        
        lgb_pipe.fit(X, y, **fit_params_lgb)
    except Exception as e:
        if args.use_gpu:
            print(f"[WARN] LightGBM GPU failed ({e}). Retrying on CPU...")
            lgb_model = get_lgbm_model(hp_lgb, use_gpu=False)
            lgb_pipe = Pipeline([('pre', preprocessor), ('model', lgb_model)])
            lgb_pipe.fit(X, y, **fit_params_lgb)
        else:
            raise e
    
    with open(outdir / "lgbm.pkl", "wb") as f:
        pickle.dump(lgb_pipe, f)
    
    # -------------------------------------------------------------------------
    # Train CatBoost
    # -------------------------------------------------------------------------
    print("\n[Full GBM] Training CatBoost...")
    cat_pipe = None
    
    try:
        cat_model = get_cat_model(hp_cat, args.use_gpu)
        cat_pipe = Pipeline([('pre', preprocessor), ('model', cat_model)])
        
        fit_params_cat = {}
        if w is not None:
            fit_params_cat['model__sample_weight'] = w
        
        cat_pipe.fit(X, y, **fit_params_cat)
    except Exception as e:
        error_msg = str(e).lower()
        if 'out of memory' in error_msg or 'cuda' in error_msg or 'gpu' in error_msg:
            print(f"[WARN] CatBoost GPU failed ({e}). Retrying on CPU...")
            try:
                cat_model = get_cat_model(hp_cat, use_gpu=False)
                cat_pipe = Pipeline([('pre', preprocessor), ('model', cat_model)])
                cat_pipe.fit(X, y, **fit_params_cat)
            except Exception as e2:
                print(f"[ERROR] CatBoost training failed: {e2}")
                cat_pipe = None
        else:
            print(f"[ERROR] CatBoost training failed: {e}")
            cat_pipe = None
    
    if cat_pipe:
        with open(outdir / "cat.pkl", "wb") as f:
            pickle.dump(cat_pipe, f)
    
    # -------------------------------------------------------------------------
    # Save Manifest
    # -------------------------------------------------------------------------
    manifest = {
        "features": list(X.columns),
        "n_train": len(df),
        "xgb_params": hp_xgb,
        "lgb_params": hp_lgb,
        "cat_params": hp_cat,
        "models_saved": ["xgb.pkl", "lgbm.pkl"] + (["cat.pkl"] if cat_pipe else [])
    }
    
    with open(outdir / "gbm_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n[Full GBM] Done. Models saved to {outdir}/")


if __name__ == "__main__":
    main()
