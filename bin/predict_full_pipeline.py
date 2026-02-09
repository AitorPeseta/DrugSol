#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict Full Pipeline: Production Inference
============================================

Production inference pipeline that generates predictions for new molecules
using the packaged final product model.

Inference Flow:
    1. Load test data (Mordred features + SMILES/RDKit features)
    2. Generate base model predictions:
       - XGBoost, LightGBM, CatBoost (from GBM directory)
       - Chemprop D-MPNN (from GNN directory)
       - Physics baseline (from Physics directory)
    3. Combine predictions using trained meta-learner
    4. Output final predictions

Arguments:
    --data-gbm       : Parquet with Mordred features for GBM models
    --data-gnn       : Parquet with SMILES + RDKit features
    --final-product  : Directory containing the final product package
    --output         : Output CSV filename

Usage:
    python predict_full_pipeline.py \\
        --data-gbm test_mordred.parquet \\
        --data-gnn test_smiles.parquet \\
        --final-product drugsol_model \\
        --output predictions.csv

Output:
    CSV file with columns: smiles, predicted_logS
"""

import argparse
import json
import os
import pickle
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# UTILITIES
# ============================================================================

def run_command(cmd):
    """Execute shell command with logging."""
    print(f"[Exec] CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_pickle(path):
    """Load pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def find_file(directory, pattern):
    """Find file recursively by pattern."""
    matches = list(Path(directory).rglob(pattern))
    return matches[0] if matches else None


def filter_model_features(model, df, model_name="Model"):
    """Select only the columns expected by the model."""
    if hasattr(model, "feature_names_in_"):
        required_cols = model.feature_names_in_
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0
        return df[required_cols]
    
    # Fallback if no metadata
    metadata_cols = ["row_uid", "smiles", "smiles_neutral", "logS", "InChIKey", "groups"]
    cols_to_keep = [c for c in df.columns if c not in metadata_cols]
    return df[cols_to_keep]


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_gbm(model_dir, df_raw):
    """
    Generate predictions from GBM models (XGBoost, LightGBM, CatBoost).
    
    Returns:
        Tuple of (y_xgb, y_lgbm, y_cat) prediction arrays
    """
    y_xgb = None
    y_lgbm = None
    y_cat = None
    
    # Find models
    xgb_path = find_file(model_dir / "gbm", "xgb.pkl")
    lgbm_path = find_file(model_dir / "gbm", "lgbm.pkl")
    cat_path = find_file(model_dir / "gbm", "cat.pkl")
    
    # XGBoost
    if xgb_path:
        print(f"[Exec] Predicting with XGBoost ({xgb_path.name})...")
        model = load_pickle(xgb_path)
        X_clean = filter_model_features(model, df_raw.copy(), "XGBoost")
        y_xgb = model.predict(X_clean)
    
    # LightGBM
    if lgbm_path:
        print(f"[Exec] Predicting with LightGBM ({lgbm_path.name})...")
        model = load_pickle(lgbm_path)
        X_clean = filter_model_features(model, df_raw.copy(), "LightGBM")
        y_lgbm = model.predict(X_clean)
    
    # CatBoost
    if cat_path:
        print(f"[Exec] Predicting with CatBoost ({cat_path.name})...")
        model = load_pickle(cat_path)
        X_clean = filter_model_features(model, df_raw.copy(), "CatBoost")
        y_cat = model.predict(X_clean)
    
    # Handle missing models with fallbacks
    n_samples = len(df_raw)
    if y_xgb is None and y_lgbm is None and y_cat is None:
        print("[WARN] No GBM models found. Returning zeros.")
        return np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)
    
    # Fill missing with available predictions
    available = [p for p in [y_xgb, y_lgbm, y_cat] if p is not None]
    fallback = available[0]
    
    if y_xgb is None:
        y_xgb = fallback
    if y_lgbm is None:
        y_lgbm = fallback
    if y_cat is None:
        y_cat = fallback
    
    return y_xgb, y_lgbm, y_cat


def predict_physics(model_dir, df_raw):
    """
    Generate predictions from physics-informed model.
    
    Supports both JSON (explicit coefficients) and Pickle (sklearn) formats.
    """
    physics_dir = model_dir / "physics"
    
    # Try JSON format first
    json_candidates = list(physics_dir.glob("*.json")) + list(physics_dir.glob("input_physics"))
    json_candidates = [f for f in json_candidates if f.is_file()]
    
    if json_candidates:
        model_path = json_candidates[0]
        print(f"[Exec] Predicting with Physics JSON ({model_path.name})...")
        
        try:
            with open(model_path, 'r') as f:
                model = json.load(f)
            
            # Prepare physical features
            df_calc = df_raw.copy()
            if "temp_C" in df_calc.columns:
                df_calc["inv_temp"] = 1000.0 / (df_calc["temp_C"] + 273.15)
            else:
                df_calc["inv_temp"] = 1000.0 / 298.15  # Default 25°C
            
            # Calculate prediction
            intercept = model.get("intercept", 0.0)
            preds = np.full(len(df_calc), intercept)
            
            # Support both "coefs" and "coefficients" keys
            coefs = model.get("coefficients", model.get("coefs", {}))
            
            for feat, weight in coefs.items():
                val = None
                # Try exact match
                if feat in df_calc.columns:
                    val = df_calc[feat]
                # Try without rdkit__ prefix
                elif feat.replace("rdkit__", "") in df_calc.columns:
                    val = df_calc[feat.replace("rdkit__", "")]
                
                if val is not None:
                    preds += pd.to_numeric(val, errors='coerce').fillna(0.0).values * weight
            
            return preds
            
        except Exception as e:
            print(f"[WARN] JSON physics model failed: {e}. Trying pickle...")
    
    # Fallback to pickle
    pkl_files = list(physics_dir.rglob("*.pkl"))
    model_path = next((p for p in pkl_files if "meta" not in p.name), None)
    
    if not model_path:
        print(f"[WARN] No physics model found in {physics_dir}. Returning zeros.")
        return np.zeros(len(df_raw))
    
    print(f"[Exec] Predicting with Physics Pickle ({model_path.name})...")
    model = load_pickle(model_path)
    
    # Handle legacy column naming
    df_mapped = df_raw.copy()
    rename_map = {c: c.replace("rdkit__", "") for c in df_mapped.columns if c.startswith("rdkit__")}
    if rename_map:
        df_mapped = df_mapped.rename(columns=rename_map)
    
    X_clean = filter_model_features(model, df_mapped, "Physics")
    return model.predict(X_clean)


def predict_gnn(model_dir, input_csv, output_csv, batch_size=50):
    """Generate predictions from Chemprop GNN model."""
    print("[Exec] Predicting with Chemprop (GNN)...")
    
    checkpoint_dir = model_dir / "gnn"
    model_path = find_file(checkpoint_dir, "*.pt")
    
    if not model_path:
        print(f"[WARN] No GNN model (.pt) found in {checkpoint_dir}. Returning zeros.")
        return np.zeros(pd.read_csv(input_csv).shape[0])
    
    cmd = [
        "chemprop_predict",
        "--test_path", str(input_csv),
        "--preds_path", str(output_csv),
        "--checkpoint_path", str(model_path),
        "--batch_size", str(batch_size),
        "--smiles_columns", "smiles_neutral"
    ]
    
    try:
        run_command(cmd)
        df_res = pd.read_csv(output_csv)
        pred_cols = [c for c in df_res.columns if c not in ["smiles_neutral", "smiles"]]
        return df_res[pred_cols[0]].values
    except Exception as e:
        print(f"[WARN] Chemprop failed: {e}. Returning zeros.")
        return np.zeros(pd.read_csv(input_csv).shape[0])


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for production inference."""
    
    ap = argparse.ArgumentParser(
        description="Generate predictions using final product model."
    )
    ap.add_argument("--data-gbm", required=True,
                    help="Parquet with Mordred features")
    ap.add_argument("--data-gnn", required=True,
                    help="Parquet with SMILES + RDKit features")
    ap.add_argument("--final-product", required=True,
                    help="Final product directory")
    ap.add_argument("--output", default="predictions.csv",
                    help="Output CSV filename")
    
    args = ap.parse_args()
    
    prod_dir = Path(args.final_product)
    base_models_dir = prod_dir / "base_models"
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("[Exec] Loading input data...")
    df_gbm = pd.read_parquet(args.data_gbm)
    df_gnn = pd.read_parquet(args.data_gnn)
    
    print(f"[Exec] Samples: {len(df_gbm):,}")
    
    # -------------------------------------------------------------------------
    # Generate Base Predictions
    # -------------------------------------------------------------------------
    y_xgb, y_lgbm, y_cat = predict_gbm(base_models_dir, df_gbm)
    pred_physics = predict_physics(base_models_dir, df_gnn)
    
    # GNN requires CSV input
    tmp_smiles = "temp_gnn_input.csv"
    tmp_preds = "temp_gnn_output.csv"
    
    if "smiles_neutral" not in df_gnn.columns:
        df_gnn["smiles_neutral"] = df_gnn.get("smiles", df_gnn.iloc[:, 0])
    
    df_gnn[["smiles_neutral"]].to_csv(tmp_smiles, index=False)
    pred_gnn = predict_gnn(base_models_dir, tmp_smiles, tmp_preds)
    
    # Cleanup temp files
    for f in [tmp_smiles, tmp_preds]:
        if os.path.exists(f):
            os.remove(f)
    
    # -------------------------------------------------------------------------
    # Ensemble Combination
    # -------------------------------------------------------------------------
    print("[Exec] Combining predictions with meta-learner...")
    
    # Build meta-features DataFrame with multiple name aliases
    X_meta = pd.DataFrame({
        "y_xgb": y_xgb,
        "y_lgbm": y_lgbm,
        "y_cat": y_cat,
        "y_physics": pred_physics,
        "y_chemprop": pred_gnn,
        "y_gnn": pred_gnn,  # Legacy alias
        # OOF-style aliases for compatibility
        "oof_xgb": y_xgb,
        "oof_lgbm": y_lgbm,
        "oof_cat": y_cat,
        "oof_physics": pred_physics,
        "oof_gnn": pred_gnn,
        "oof_chemprop": pred_gnn
    })
    
    final_pred = None
    
    # Try loading stacker
    meta_model_path = prod_dir / "meta_ridge.pkl"
    weights_path = prod_dir / "weights.json"
    
    if meta_model_path.exists():
        print("[Exec] Using stacking meta-model...")
        with open(meta_model_path, "rb") as f:
            stack_model = pickle.load(f)
        
        if isinstance(stack_model, dict) and "feature_names" in stack_model:
            required_cols = stack_model["feature_names"]
            print(f"[Exec] Stacker expects: {required_cols}")
            
            # Fill missing columns
            for c in required_cols:
                if c not in X_meta.columns:
                    print(f"[WARN] Missing column: {c}. Filling with 0.")
                    X_meta[c] = 0.0
            
            X_final = X_meta[required_cols].values
            coef = np.array(stack_model["coef"])
            intercept = stack_model["intercept"]
            final_pred = np.dot(X_final, coef.T) + intercept
        
        elif hasattr(stack_model, "predict"):
            if hasattr(stack_model, "feature_names_in_"):
                X_final = X_meta[stack_model.feature_names_in_]
            else:
                cols_fallback = sorted(["y_xgb", "y_lgbm", "y_cat", "y_physics", "y_chemprop"])
                cols_fallback = [c for c in cols_fallback if c in X_meta.columns]
                X_final = X_meta[cols_fallback]
            final_pred = stack_model.predict(X_final)
    
    elif weights_path.exists():
        print("[Exec] Using blending weights...")
        with open(weights_path, "r") as f:
            weights = json.load(f)
        
        # Build weighted sum
        final_pred = np.zeros(len(X_meta))
        total_weight = 0
        
        for col, weight in weights.items():
            if col in X_meta.columns and weight > 0:
                final_pred += X_meta[col].values * weight
                total_weight += weight
        
        if total_weight > 0:
            final_pred /= total_weight
    
    else:
        print("[WARN] No meta-model found. Using simple average.")
        cols = ["y_xgb", "y_lgbm", "y_cat", "y_chemprop", "y_physics"]
        final_pred = X_meta[cols].mean(axis=1).values
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    df_out = pd.DataFrame({
        "smiles": df_gnn["smiles_neutral"],
        "predicted_logS": final_pred
    })
    
    df_out.to_csv(args.output, index=False)
    print(f"[Exec] SUCCESS! Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
