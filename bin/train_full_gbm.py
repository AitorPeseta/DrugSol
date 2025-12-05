#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_gbm.py
-----------------
Retrains XGBoost and LightGBM on the full dataset.
Updated for XGBoost 2.0+ (tree_method='hist' + device='cuda').
Now includes SHAP Analysis for interpretability.
"""

import argparse
import json
import sys
import pickle
import statistics
import numpy as np
import pandas as pd
from pathlib import Path

# Nuevas importaciones para SHAP
import shap
import matplotlib.pyplot as plt

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
    """Aggregates params from multiple fold JSONs."""
    p = Path(hp_dir)
    # Si es un archivo directo (el consolidado), lo leemos directamente
    if p.is_file():
        try:
            full_params = json.loads(p.read_text())
            # A veces el consolidado tiene claves "xgb_params" o "lgb_params" si es un manifest
            # O puede ser el json directo de params. Asumimos estructura simple o filtrado.
            # Adaptación para tu caso actual si ya viene limpio:
            return full_params
        except:
            return {}

    # Si es directorio, lógica antigua de agregación
    files = sorted(p.glob(f"{prefix}_fold*.json"))
    if not files: return {}
    
    all_params = [json.loads(f.read_text()) for f in files]
    if not all_params: return {}
    
    keys = all_params[0].keys()
    agg = {}
    
    for k in keys:
        vals = [d[k] for d in all_params if k in d]
        if not vals: continue
        if isinstance(vals[0], float):
            agg[k] = float(np.median(vals))
        elif isinstance(vals[0], int) and not isinstance(vals[0], bool):
             agg[k] = int(round(np.median(vals)))
        else:
            try:
                agg[k] = statistics.mode(vals)
            except:
                agg[k] = vals[0]
    return agg

def get_pipeline_xgb(params, use_gpu):
    # Base
    p = {
        'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 8,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE, 'n_jobs': 4, 'verbosity': 0
    }
    # GPU Config (XGBoost 2.0+)
    if use_gpu:
        p.update({'tree_method': 'hist', 'device': 'cuda'})
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

    print("[Full GBM] Loading data...")
    df = read_any(args.train)
    y = df[args.target].values
    
    if args.sample_weight_col and args.sample_weight_col in df.columns:
        w = df[args.sample_weight_col].fillna(1.0).values
    else:
        w = None

    drop_cols = [args.target, args.sample_weight_col, "row_uid", "fold", "smiles_neutral"]
    # Guardamos columns originales para reconstruir nombres después
    X_original_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=np.number)
    X = X_original_df.copy()
    
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('var_thresh', VarianceThreshold(threshold=0.0))
    ])

    hp_xgb = get_aggregated_params(args.hp_dir, "xgb")
    hp_lgb = get_aggregated_params(args.hp_dir, "lgbm")
    
    # ---------------------------------------------------------
    # 1. Train XGBoost
    # ---------------------------------------------------------
    print("[Full GBM] Training XGBoost...")
    xgb_model = get_pipeline_xgb(hp_xgb, args.use_gpu)
    xgb_pipe = Pipeline([('pre', preprocessor), ('model', xgb_model)])
    fit_params = {}
    if w is not None: fit_params['model__sample_weight'] = w
    xgb_pipe.fit(X, y, **fit_params)
    
    with open(outdir / "xgb.pkl", "wb") as f:
        pickle.dump(xgb_pipe, f)

    # =========================================================
    # NUEVO: SHAP ANALYSIS (Solo para XGBoost)
    # =========================================================
    try:
        print("[Full GBM] Generating SHAP explanation for XGBoost...")
        
        # A. Recuperar el modelo interno y el preprocesador
        inner_model = xgb_pipe.named_steps['model']
        inner_pre = xgb_pipe.named_steps['pre']
        
        # B. Transformar X tal como lo ve el modelo (Imputación + Selección)
        # Esto devuelve un numpy array sin nombres
        X_transformed = inner_pre.transform(X)
        
        # C. Recuperar nombres de las features
        # VarianceThreshold elimina columnas, necesitamos saber cuáles quedaron
        support_mask = inner_pre.named_steps['var_thresh'].get_support()
        original_feats = X_original_df.columns
        final_feats = original_feats[support_mask]
        
        # D. Crear DataFrame con nombres para que SHAP los pinte bonitos
        X_shap = pd.DataFrame(X_transformed, columns=final_feats)
        
        # E. Calcular SHAP
        # TreeExplainer es muy rápido para XGBoost
        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer.shap_values(X_shap)
        
        # F. Plot
        plt.figure(figsize=(10, 12))
        # Summary Plot (tipo "dot" es el de abejas, el más informativo)
        shap.summary_plot(shap_values, X_shap, max_display=20, show=False, plot_type="dot")
        
        shap_out = outdir / "shap_summary_xgb.png"
        plt.savefig(shap_out, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[Full GBM] SHAP plot saved to {shap_out}")
        
    except Exception as e:
        print(f"[WARN] Could not generate SHAP plot: {e}")
    # =========================================================


    # ---------------------------------------------------------
    # 2. Train LightGBM
    # ---------------------------------------------------------
    print("[Full GBM] Training LightGBM...")
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