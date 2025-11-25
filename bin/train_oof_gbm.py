#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_oof_gbm.py
----------------
Trains XGBoost and LightGBM models using Pre-computed Folds.
Features:
- Nested Cross-Validation (Inner loop for Optuna, Outer loop for OOF).
- GPU Acceleration (Compatible with XGBoost 2.0+).
- Sample Weighting (Gaussian weights).
- ASHA Pruning for efficient tuning.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb

# Optional Optuna
try:
    import optuna
    from optuna.exceptions import TrialPruned
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    TrialPruned = Exception

RANDOM_STATE = 42

# -------------------- ARGS --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Train GBM OOF models.")
    ap.add_argument("--train", required=True, help="Train parquet/csv")
    ap.add_argument("--folds", required=True, help="Folds parquet")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--sample-weight-col", default=None)
    ap.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration")
    ap.add_argument("--save-dir", default="./oof_gbm")
    
    # Tuning
    ap.add_argument("--tune-trials", type=int, default=0, help="0 = No tuning")
    ap.add_argument("--inner-splits", type=int, default=3)
    
    # ASHA Pruner
    ap.add_argument("--pruner", default="asha")
    ap.add_argument("--asha-min-resource", type=int, default=1)
    ap.add_argument("--asha-reduction-factor", type=int, default=3)
    ap.add_argument("--asha-min-early-stopping-rate", type=int, default=0)
    
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    return ap.parse_args()

# -------------------- UTILS --------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_data(train_path, folds_path, id_col, target_col, weight_col=None):
    print(f"[Load] Reading data...")
    df = pd.read_parquet(train_path) if str(train_path).endswith('.parquet') else pd.read_csv(train_path)
    folds = pd.read_parquet(folds_path) if str(folds_path).endswith('.parquet') else pd.read_csv(folds_path)
    
    # Merge folds
    if id_col not in df.columns or id_col not in folds.columns:
        sys.exit(f"[ERROR] ID col '{id_col}' missing.")
        
    df = df.merge(folds[[id_col, 'fold']], on=id_col, how='inner')
    
    # Extract parts
    y = df[target_col].values
    ids = df[id_col].values
    fold_ids = df['fold'].values
    
    # Weights
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].fillna(1.0).values
    else:
        weights = np.ones(len(df))
        
    # Numeric Features only
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col, 'fold'], errors='ignore')
    if weight_col in X.columns: X = X.drop(columns=[weight_col])
    
    # Simple Imputation (Median) for safety
    imputer = SimpleImputer(strategy='median')
    X_np = imputer.fit_transform(X)
    X = pd.DataFrame(X_np, columns=X.columns)
    
    print(f"[Load] Shape: {X.shape}, Folds: {len(np.unique(fold_ids))}")
    return X, y, weights, ids, fold_ids

# -------------------- MODEL WRAPPERS --------------------

def train_xgb(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, use_gpu=False):
    """Trains XGBoost with Early Stopping."""
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    
    # Base params
    p = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': RANDOM_STATE,
        'nthread': 4
    }
    
    # GPU Config (XGBoost 2.0+ syntax)
    # 'gpu_hist' is deprecated. Use 'hist' + device='cuda'
    if use_gpu:
        p.update({'tree_method': 'hist', 'device': 'cuda'})
    else:
        p.update({'tree_method': 'hist', 'device': 'cpu'})
        
    # Update with hyperparams
    p.update(params)
    
    # Train
    model = xgb.train(
        p, dtrain,
        num_boost_round=3000,
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    return model

def train_lgbm(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, use_gpu=False):
    """Trains LightGBM with Early Stopping and fallback."""
    
    p = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': RANDOM_STATE,
        'n_jobs': 4,
        'n_estimators': 3000
    }
    
    # GPU Config attempt
    if use_gpu:
        p.update({'device_type': 'gpu'})
        
    p.update(params)
    
    model = lgb.LGBMRegressor(**p)
    
    try:
        model.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[w_va],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
    except Exception as e:
        if use_gpu:
            print(f"[WARN] LightGBM GPU failed ({e}). Retrying on CPU...")
            p['device_type'] = 'cpu'
            model = lgb.LGBMRegressor(**p)
            model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                eval_sample_weight=[w_va],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        else:
            raise e
            
    return model

# -------------------- OPTUNA --------------------

def get_space_xgb(trial):
    return {
        # CAMBIO CLAVE: Mínimo 0.05 en vez de 0.001
        'learning_rate': trial.suggest_float('lr', 0.05, 0.3, log=True), 
        'max_depth': trial.suggest_int('depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child', 1, 20)
    }

def get_space_lgbm(trial):
    return {
        # CAMBIO CLAVE: Mínimo 0.05
        'learning_rate': trial.suggest_float('lr', 0.05, 0.3, log=True),
        'num_leaves': trial.suggest_int('leaves', 20, 128), # Reducido max de 256 a 128
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child', 10, 100)
    }

def optimize_fold(algo, X, y, w, train_idx, fold_idx, args):
    """Optimizes hyperparameters for a specific fold using Inner CV."""
    
    # Inner Split for Tuning
    kf = KFold(n_splits=args.inner_splits, shuffle=True, random_state=RANDOM_STATE)
    
    # Cache dataset slices to avoid pandas overhead in trials
    X_tr_outer = X.iloc[train_idx]
    y_tr_outer = y[train_idx]
    w_tr_outer = w[train_idx]
    
    def objective(trial):
        params = get_space_xgb(trial) if algo == 'xgb' else get_space_lgbm(trial)
        scores = []
        
        for i, (inner_tr, inner_va) in enumerate(kf.split(X_tr_outer)):
            # Inner Train/Val
            X_it, X_iv = X_tr_outer.iloc[inner_tr], X_tr_outer.iloc[inner_va]
            y_it, y_iv = y_tr_outer[inner_tr], y_tr_outer[inner_va]
            w_it, w_iv = w_tr_outer[inner_tr], w_tr_outer[inner_va]
            
            if algo == 'xgb':
                m = train_xgb(X_it, y_it, w_it, X_iv, y_iv, w_iv, params, args.use_gpu)
                preds = m.predict(xgb.DMatrix(X_iv))
            else:
                m = train_lgbm(X_it, y_it, w_it, X_iv, y_iv, w_iv, params, args.use_gpu)
                preds = m.predict(X_iv)
                
            scores.append(rmse(y_iv, preds))
            
            # Pruning check
            trial.report(np.mean(scores), i)
            if trial.should_prune():
                raise TrialPruned()
                
        return np.mean(scores)

    pruner = optuna.pruners.SuccessiveHalvingPruner() if args.pruner == 'asha' else optuna.pruners.NopPruner()
    
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=args.tune_trials, show_progress_bar=False)
    
    return study.best_params

# -------------------- MAIN --------------------

def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    (save_dir / "oof").mkdir(parents=True, exist_ok=True)
    (save_dir / "hp").mkdir(parents=True, exist_ok=True)
    
    X, y, w, ids, folds = load_data(args.train, args.folds, args.id_col, args.target, args.sample_weight_col)
    unique_folds = np.unique(folds)
    
    # Storage for predictions
    oof_preds_xgb = np.zeros(len(y))
    oof_preds_lgbm = np.zeros(len(y))
    
    metrics = {"xgb": {}, "lgbm": {}}
    
    print(f"[Train] Starting OOF loop for {len(unique_folds)} folds...")
    
    for fold in unique_folds:
        print(f"  > Fold {fold}")
        
        tr_idx = folds != fold
        va_idx = folds == fold
        
        X_tr, y_tr, w_tr = X.iloc[tr_idx], y[tr_idx], w[tr_idx]
        X_va, y_va, w_va = X.iloc[va_idx], y[va_idx], w[va_idx]
        
        # --- XGBoost ---
        if args.tune_trials > 0:
            best_xgb = optimize_fold('xgb', X, y, w, tr_idx, fold, args)
            # Save params
            with open(save_dir / "hp" / f"xgb_fold{fold}.json", 'w') as f:
                json.dump(best_xgb, f, indent=2)
        else:
            best_xgb = {'learning_rate': 0.03, 'max_depth': 8} # Default
            
        model_xgb = train_xgb(X_tr, y_tr, w_tr, X_va, y_va, w_va, best_xgb, args.use_gpu)
        oof_preds_xgb[va_idx] = model_xgb.predict(xgb.DMatrix(X_va))
        
        # --- LightGBM ---
        if args.tune_trials > 0:
            best_lgbm = optimize_fold('lgbm', X, y, w, tr_idx, fold, args)
            with open(save_dir / "hp" / f"lgbm_fold{fold}.json", 'w') as f:
                json.dump(best_lgbm, f, indent=2)
        else:
            best_lgbm = {'learning_rate': 0.03, 'num_leaves': 128}
            
        model_lgbm = train_lgbm(X_tr, y_tr, w_tr, X_va, y_va, w_va, best_lgbm, args.use_gpu)
        oof_preds_lgbm[va_idx] = model_lgbm.predict(X_va)

    # Save Results
    df_xgb = pd.DataFrame({'id': ids, 'fold': folds, 'y_true': y, 'y_pred': oof_preds_xgb})
    df_lgbm = pd.DataFrame({'id': ids, 'fold': folds, 'y_true': y, 'y_pred': oof_preds_lgbm})
    
    df_xgb.to_parquet(save_dir / "oof" / "xgb.parquet", index=False)
    df_lgbm.to_parquet(save_dir / "oof" / "lgbm.parquet", index=False)
    
    # Metrics
    metrics['xgb']['rmse'] = rmse(y, oof_preds_xgb)
    metrics['xgb']['r2'] = r2_score(y, oof_preds_xgb)
    metrics['lgbm']['rmse'] = rmse(y, oof_preds_lgbm)
    metrics['lgbm']['r2'] = r2_score(y, oof_preds_lgbm)
    
    with open(save_dir / "metrics_tree.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[Done] XGB RMSE: {metrics['xgb']['rmse']:.4f} | LGBM RMSE: {metrics['lgbm']['rmse']:.4f}")

if __name__ == "__main__":
    main()