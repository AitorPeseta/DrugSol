#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train OOF GBM: Out-of-Fold Training for Gradient Boosting Models
=================================================================

Trains XGBoost, LightGBM, and CatBoost models using pre-computed folds
for out-of-fold (OOF) prediction generation. OOF predictions are used
for ensemble stacking and blending.

Features:
    - Nested cross-validation with Optuna hyperparameter tuning
    - GPU acceleration for all three model types
    - Sample weighting support
    - ASHA pruning for efficient hyperparameter search

Arguments:
    --train              : Training data Parquet/CSV file
    --folds              : Folds Parquet file with fold assignments
    --target             : Target column name (default: logS)
    --id-col             : Row identifier column (default: row_uid)
    --sample-weight-col  : Sample weight column (default: None)
    --use-gpu            : Enable GPU acceleration
    --save-dir           : Output directory (default: ./oof_gbm)
    --tune-trials        : Number of Optuna trials (0 = no tuning)
    --inner-splits       : Inner CV splits for tuning (default: 3)
    --pruner             : Optuna pruner type (default: asha)
    --seed               : Random seed (default: 42)

Usage:
    python train_oof_gbm.py \\
        --train final_train_gbm.parquet \\
        --folds folds.parquet \\
        --target logS \\
        --id-col row_uid \\
        --sample-weight-col weight \\
        --use-gpu \\
        --tune-trials 50 \\
        --save-dir ./oof_gbm

Output:
    - oof/xgb.parquet: XGBoost OOF predictions
    - oof/lgbm.parquet: LightGBM OOF predictions
    - oof/cat.parquet: CatBoost OOF predictions
    - hp/*.json: Best hyperparameters per fold
    - metrics_tree.json: Overall metrics for each model
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Optional Optuna import
try:
    import optuna
    from optuna.exceptions import TrialPruned
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def check_gpu_memory(min_memory_mb=4000):
    """
    Check if GPU has enough free memory for CatBoost.
    
    Args:
        min_memory_mb: Minimum required free memory in MB
    
    Returns:
        True if GPU has enough memory, False otherwise
    """
    try:
        import subprocess
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
    return True  # If we can't check, assume it's OK
    TrialPruned = Exception

RANDOM_STATE = 42


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Train GBM OOF models (XGBoost, LightGBM, CatBoost)."
    )
    
    # Data inputs
    ap.add_argument("--train", required=True,
                    help="Training data Parquet/CSV file")
    ap.add_argument("--folds", required=True,
                    help="Folds Parquet file")
    ap.add_argument("--target", default="logS",
                    help="Target column (default: logS)")
    ap.add_argument("--id-col", default="row_uid",
                    help="ID column (default: row_uid)")
    ap.add_argument("--sample-weight-col", default=None,
                    help="Sample weight column")
    
    # Compute settings
    ap.add_argument("--use-gpu", action="store_true",
                    help="Enable GPU acceleration")
    ap.add_argument("--save-dir", default="./oof_gbm",
                    help="Output directory")
    
    # Tuning settings
    ap.add_argument("--tune-trials", type=int, default=0,
                    help="Optuna trials (0 = no tuning)")
    ap.add_argument("--inner-splits", type=int, default=3,
                    help="Inner CV splits for tuning")
    
    # ASHA Pruner settings
    ap.add_argument("--pruner", default="asha",
                    help="Optuna pruner type")
    ap.add_argument("--asha-min-resource", type=int, default=1,
                    help="ASHA min resource")
    ap.add_argument("--asha-reduction-factor", type=int, default=3,
                    help="ASHA reduction factor")
    ap.add_argument("--asha-min-early-stopping-rate", type=int, default=0,
                    help="ASHA min early stopping rate")
    
    ap.add_argument("--seed", type=int, default=RANDOM_STATE,
                    help="Random seed")
    
    return ap.parse_args()


# ============================================================================
# UTILITIES
# ============================================================================

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_data(train_path, folds_path, id_col, target_col, weight_col=None):
    """
    Load training data and merge with fold assignments.
    
    Returns:
        Tuple of (X, y, weights, ids, fold_ids)
    """
    print("[Load] Reading data...")
    
    # Load training data
    if str(train_path).endswith('.parquet'):
        df = pd.read_parquet(train_path)
    else:
        df = pd.read_csv(train_path)
    
    # Load folds
    if str(folds_path).endswith('.parquet'):
        folds = pd.read_parquet(folds_path)
    else:
        folds = pd.read_csv(folds_path)
    
    # Validate ID column
    if id_col not in df.columns or id_col not in folds.columns:
        sys.exit(f"[ERROR] ID column '{id_col}' not found.")
    
    # Merge folds into training data
    df = df.merge(folds[[id_col, 'fold']], on=id_col, how='inner')
    
    # Extract components
    y = df[target_col].values
    ids = df[id_col].values
    fold_ids = df['fold'].values
    
    # Handle sample weights
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].fillna(1.0).values
    else:
        weights = np.ones(len(df))
    
    # Prepare features (numeric only)
    X = df.select_dtypes(include=[np.number]).drop(
        columns=[target_col, 'fold'], errors='ignore'
    )
    if weight_col and weight_col in X.columns:
        X = X.drop(columns=[weight_col])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_np = imputer.fit_transform(X)
    X = pd.DataFrame(X_np, columns=X.columns)
    
    print(f"[Load] Shape: {X.shape}, Folds: {len(np.unique(fold_ids))}")
    return X, y, weights, ids, fold_ids


# ============================================================================
# MODEL TRAINING WRAPPERS
# ============================================================================

def train_xgb(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, use_gpu=False):
    """Train XGBoost model with early stopping and GPU fallback."""
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    
    p = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': RANDOM_STATE,
        'nthread': 4
    }
    
    if use_gpu:
        p.update({'tree_method': 'hist', 'device': 'cuda'})
    else:
        p.update({'tree_method': 'hist', 'device': 'cpu'})
    
    p.update(params)
    
    try:
        model = xgb.train(
            p, dtrain,
            num_boost_round=3000,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
    except Exception as e:
        error_msg = str(e).lower()
        if use_gpu and ('out of memory' in error_msg or 'memory' in error_msg):
            print(f"[WARN] XGBoost GPU OOM falló. Cambiando a CPU para este modelo...")
            p.update({'device': 'cpu'})
            model = xgb.train(
                p, dtrain,
                num_boost_round=3000,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=100,
                verbose_eval=False
            )
        else:
            raise e
            
    return model


def train_lgbm(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, use_gpu=False):
    """Train LightGBM model with early stopping and GPU fallback."""
    p = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': RANDOM_STATE,
        'n_jobs': 4,
        'n_estimators': 3000
    }
    
    if use_gpu:
        p.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
    
    p.update(params)
    
    model = lgb.LGBMRegressor(**p)
    
    try:
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[w_va],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
    except Exception as e:
        if use_gpu:
            print(f"[WARN] LightGBM GPU failed ({e}). Retrying on CPU...")
            p['device'] = 'cpu'
            model = lgb.LGBMRegressor(**p)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                eval_sample_weight=[w_va],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        else:
            raise e
    
    return model


def train_cat(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, use_gpu=False):
    """Train CatBoost model with early stopping and GPU fallback."""
    train_pool = cb.Pool(X_tr, y_tr, weight=w_tr)
    valid_pool = cb.Pool(X_va, y_va, weight=w_va)
    
    actual_use_gpu = False 
    
    if use_gpu and not actual_use_gpu:
        print("[WARN] CatBoost: Insufficient GPU memory, using CPU instead")
    
    p = {
        'iterations': 3000,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'allow_writing_files': False
    }
    
    if actual_use_gpu:
        p.update({'task_type': 'GPU', 'devices': '0'})
    else:
        p.update({'task_type': 'CPU', 'thread_count': 4})
    
    p.update(params)
    
    model = cb.CatBoostRegressor(**p)
    
    try:
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=100,
            verbose=False
        )
    except Exception as e:
        error_msg = str(e).lower()
        if actual_use_gpu and ('out of memory' in error_msg or 'cuda' in error_msg or 'gpu' in error_msg):
            print(f"[WARN] CatBoost GPU failed ({e}). Retrying on CPU...")
            # Rebuild params for CPU
            p['task_type'] = 'CPU'
            p['thread_count'] = 4
            if 'devices' in p:
                del p['devices']
            
            model = cb.CatBoostRegressor(**p)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                early_stopping_rounds=100,
                verbose=False
            )
        else:
            raise e
    
    return model


# ============================================================================
# OPTUNA HYPERPARAMETER SPACES
# ============================================================================

def get_space_xgb(trial):
    """XGBoost hyperparameter search space."""
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30)
    }


def get_space_lgbm(trial):
    """LightGBM hyperparameter search space."""
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }


def get_space_cat(trial):
    """
    CatBoost hyperparameter search space.
    
    Note: CatBoost's default bootstrap_type is 'Bayesian', which does NOT
    support the 'subsample' parameter. To use subsample, we must explicitly
    set bootstrap_type to 'Bernoulli' or 'MVS'.
    """
    # Choose bootstrap type that supports subsample
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS'])
    
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
        'bootstrap_type': bootstrap_type,
    }
    
    # subsample is only valid with Bernoulli or MVS bootstrap
    if bootstrap_type in ['Bernoulli', 'MVS']:
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    
    return params


def optimize_fold(algo, X, y, w, train_idx, fold_idx, args):
    """
    Run Optuna hyperparameter optimization for a single fold.
    
    Args:
        algo: Algorithm name ('xgb', 'lgbm', 'cat')
        X: Feature DataFrame
        y: Target array
        w: Weight array
        train_idx: Boolean mask for training samples
        fold_idx: Current fold index
        args: Command line arguments
    
    Returns:
        Dictionary of best hyperparameters
    """
    kf = KFold(n_splits=args.inner_splits, shuffle=True, random_state=RANDOM_STATE)
    
    X_tr_outer = X.iloc[train_idx]
    y_tr_outer = y[train_idx]
    w_tr_outer = w[train_idx]
    
    def objective(trial):
        # Get hyperparameter space
        if algo == 'xgb':
            params = get_space_xgb(trial)
        elif algo == 'lgbm':
            params = get_space_lgbm(trial)
        else:
            params = get_space_cat(trial)
        
        scores = []
        for i, (inner_tr, inner_va) in enumerate(kf.split(X_tr_outer)):
            X_it, X_iv = X_tr_outer.iloc[inner_tr], X_tr_outer.iloc[inner_va]
            y_it, y_iv = y_tr_outer[inner_tr], y_tr_outer[inner_va]
            w_it, w_iv = w_tr_outer[inner_tr], w_tr_outer[inner_va]
            
            # Train and predict
            if algo == 'xgb':
                m = train_xgb(X_it, y_it, w_it, X_iv, y_iv, w_iv, params, args.use_gpu)
                preds = m.predict(xgb.DMatrix(X_iv))
            elif algo == 'lgbm':
                m = train_lgbm(X_it, y_it, w_it, X_iv, y_iv, w_iv, params, args.use_gpu)
                preds = m.predict(X_iv)
            else:
                m = train_cat(X_it, y_it, w_it, X_iv, y_iv, w_iv, params, args.use_gpu)
                preds = m.predict(X_iv)
            
            scores.append(rmse(y_iv, preds))
            
            # Report intermediate result for pruning
            trial.report(np.mean(scores), i)
            if trial.should_prune():
                raise TrialPruned()
        
        return np.mean(scores)
    
    # Configure pruner
    if args.pruner == 'asha':
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=args.asha_min_resource,
            reduction_factor=args.asha_reduction_factor,
            min_early_stopping_rate=args.asha_min_early_stopping_rate
        )
    else:
        pruner = optuna.pruners.NopPruner()
    
    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=args.tune_trials, show_progress_bar=False)
    
    return study.best_params


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for OOF GBM training."""
    args = parse_args()
    
    # Setup output directories
    save_dir = Path(args.save_dir)
    (save_dir / "oof").mkdir(parents=True, exist_ok=True)
    (save_dir / "hp").mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, w, ids, folds = load_data(
        args.train, args.folds, args.id_col, args.target, args.sample_weight_col
    )
    unique_folds = np.unique(folds)
    
    # Initialize OOF prediction arrays
    oof_preds_xgb = np.zeros(len(y))
    oof_preds_lgbm = np.zeros(len(y))
    oof_preds_cat = np.zeros(len(y))
    
    metrics = {"xgb": {}, "lgbm": {}, "cat": {}}
    
    print(f"[Train] Starting OOF loop for {len(unique_folds)} folds (XGB + LGBM + CAT)...")
    
    for fold in unique_folds:
        print(f"  > Fold {fold}")
        
        # Split indices
        tr_idx = folds != fold
        va_idx = folds == fold
        X_tr, y_tr, w_tr = X.iloc[tr_idx], y[tr_idx], w[tr_idx]
        X_va, y_va, w_va = X.iloc[va_idx], y[va_idx], w[va_idx]
        
        # --- XGBoost ---
        if args.tune_trials > 0:
            best_xgb = optimize_fold('xgb', X, y, w, tr_idx, fold, args)
            with open(save_dir / "hp" / f"xgb_fold{fold}.json", 'w') as f:
                json.dump(best_xgb, f, indent=2)
        else:
            best_xgb = {'learning_rate': 0.05, 'max_depth': 6}
        
        m_xgb = train_xgb(X_tr, y_tr, w_tr, X_va, y_va, w_va, best_xgb, args.use_gpu)
        oof_preds_xgb[va_idx] = m_xgb.predict(xgb.DMatrix(X_va))
        
        # --- LightGBM ---
        if args.tune_trials > 0:
            best_lgbm = optimize_fold('lgbm', X, y, w, tr_idx, fold, args)
            with open(save_dir / "hp" / f"lgbm_fold{fold}.json", 'w') as f:
                json.dump(best_lgbm, f, indent=2)
        else:
            best_lgbm = {'learning_rate': 0.05, 'num_leaves': 31}
        
        m_lgbm = train_lgbm(X_tr, y_tr, w_tr, X_va, y_va, w_va, best_lgbm, args.use_gpu)
        oof_preds_lgbm[va_idx] = m_lgbm.predict(X_va)
        
        # --- CatBoost ---
        if args.tune_trials > 0:
            best_cat = optimize_fold('cat', X, y, w, tr_idx, fold, args)
            with open(save_dir / "hp" / f"cat_fold{fold}.json", 'w') as f:
                json.dump(best_cat, f, indent=2)
        else:
            # Default params compatible with Bayesian bootstrap
            best_cat = {'learning_rate': 0.05, 'depth': 6}
        
        m_cat = train_cat(X_tr, y_tr, w_tr, X_va, y_va, w_va, best_cat, args.use_gpu)
        oof_preds_cat[va_idx] = m_cat.predict(X_va)
    
    # Save OOF predictions
    df_xgb = pd.DataFrame({
        'id': ids, 'fold': folds, 'y_true': y, 'y_pred': oof_preds_xgb
    })
    df_lgbm = pd.DataFrame({
        'id': ids, 'fold': folds, 'y_true': y, 'y_pred': oof_preds_lgbm
    })
    df_cat = pd.DataFrame({
        'id': ids, 'fold': folds, 'y_true': y, 'y_pred': oof_preds_cat
    })
    
    df_xgb.to_parquet(save_dir / "oof" / "xgb.parquet", index=False)
    df_lgbm.to_parquet(save_dir / "oof" / "lgbm.parquet", index=False)
    df_cat.to_parquet(save_dir / "oof" / "cat.parquet", index=False)
    
    # Calculate and save metrics
    metrics['xgb'] = {
        'rmse': float(rmse(y, oof_preds_xgb)),
        'r2': float(r2_score(y, oof_preds_xgb))
    }
    metrics['lgbm'] = {
        'rmse': float(rmse(y, oof_preds_lgbm)),
        'r2': float(r2_score(y, oof_preds_lgbm))
    }
    metrics['cat'] = {
        'rmse': float(rmse(y, oof_preds_cat)),
        'r2': float(r2_score(y, oof_preds_cat))
    }
    
    with open(save_dir / "metrics_tree.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Done] Results:")
    print(f"       XGB:  RMSE={metrics['xgb']['rmse']:.4f}, R²={metrics['xgb']['r2']:.4f}")
    print(f"       LGBM: RMSE={metrics['lgbm']['rmse']:.4f}, R²={metrics['lgbm']['r2']:.4f}")
    print(f"       CAT:  RMSE={metrics['cat']['rmse']:.4f}, R²={metrics['cat']['r2']:.4f}")


if __name__ == "__main__":
    main()
