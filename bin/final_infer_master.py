#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Inference Master: Ensemble Prediction Pipeline
=====================================================

Master inference pipeline that generates predictions from all trained models
and combines them using learned ensemble weights.

Inference Pipeline:
    1. Load test data (tabular features + SMILES)
    2. Generate Level-0 predictions from base models:
       - XGBoost, LightGBM, CatBoost (GBM ensemble)
       - Chemprop D-MPNN (graph neural network)
       - Physics baseline (Ridge regression)
    3. Combine predictions via:
       - Blending: Weighted average using simplex-constrained weights
       - Stacking: Meta-model using Ridge regression coefficients
    4. Calculate metrics including physiological temperature range

Arguments:
    --test-tabular       : Test data with tabular features (Parquet/CSV)
    --test-smiles        : Test data with SMILES (Parquet/CSV)
    --models-dir         : Directory with GBM models (xgb.pkl, lgbm.pkl, cat.pkl)
    --chemprop-model-dir : Directory with Chemprop model (.pt checkpoint)
    --physics-json       : Physics model JSON file
    --id-col             : Row identifier column (default: row_uid)
    --smiles-col         : SMILES column name (default: smiles_neutral)
    --target             : Target column name (default: logS)
    --save-dir           : Output directory
    --weights-json       : Blending weights JSON file
    --stack-pkl          : Stacking meta-model pickle file

Usage:
    python final_infer_master.py \\
        --test-tabular test_features.parquet \\
        --test-smiles test_smiles.parquet \\
        --models-dir models_GBM \\
        --chemprop-model-dir models_GNN \\
        --physics-json models_physics/physics_model.json \\
        --weights-json ensemble/blend/weights.json \\
        --stack-pkl ensemble/stack/meta_ridge.pkl \\
        --save-dir pred

Output:
    - test_level0.parquet: Predictions from all base models
    - test_blend.parquet: Blended ensemble predictions
    - test_stack.parquet: Stacked ensemble predictions
    - metrics_test.json: Performance metrics

Notes:
    - Missing models are skipped gracefully
    - GPU is used for Chemprop if available
    - Physiological range metrics (35-38°C) included when temperature available
"""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Optional CatBoost import
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ============================================================================
# UTILITIES
# ============================================================================

def run_command(cmd, check=True):
    """Run shell command with logging."""
    print(f"[CMD] {' '.join(map(str, cmd))}")
    try:
        subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        raise e


def read_any(path: str) -> pd.DataFrame:
    """Read Parquet or CSV file."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        if p.suffix == '.parquet':
            return pd.read_parquet(p)
        return pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] Error reading {path}: {e}")
        return pd.DataFrame()


def calc_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "r2": float(r2_score(y_true[mask], y_pred[mask])),
        "n": int(mask.sum())
    }


def normalize_id(series):
    """Normalize ID column to string."""
    return series.astype("string").str.strip()


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_chemprop(model_dir, test_df, smiles_col, id_col, gpu=False):
    """
    Generate predictions using Chemprop model.
    
    Args:
        model_dir: Directory containing Chemprop checkpoint
        test_df: Test DataFrame with SMILES
        smiles_col: SMILES column name
        id_col: ID column name
        gpu: Whether to use GPU
    
    Returns:
        DataFrame with id and y_chemprop columns
    """
    if test_df.empty:
        return pd.DataFrame(columns=[id_col, "y_chemprop"])
    
    p = Path(model_dir)
    ckpts = sorted(list(p.rglob("*.pt")))
    if not ckpts:
        print("[WARN] No Chemprop checkpoint found")
        return pd.DataFrame({id_col: test_df[id_col]})
    
    # Prefer model.pt (final checkpoint)
    ckpt = next((c for c in ckpts if c.name == "model.pt"), ckpts[0])
    
    print(f"[Infer] Chemprop (checkpoint: {ckpt.name})...")
    
    # Prepare input CSV
    tmp_in = p / "test_in.csv"
    tmp_out = p / "test_out.csv"
    
    df_in = test_df.copy()
    if smiles_col not in df_in.columns:
        print(f"[WARN] SMILES column '{smiles_col}' not found")
        return pd.DataFrame({id_col: test_df[id_col]})
    
    df_in = df_in.rename(columns={smiles_col: "smiles"})
    df_in[["smiles"]].to_csv(tmp_in, index=False)
    
    # Build command
    cmd = [
        "chemprop_predict",
        "--test_path", str(tmp_in),
        "--preds_path", str(tmp_out),
        "--checkpoint_path", str(ckpt),
        "--batch_size", "64",
        "--smiles_columns", "smiles"
    ]
    if gpu:
        cmd += ["--gpu", "0"]
    
    try:
        run_command(cmd)
        if not tmp_out.exists():
            raise FileNotFoundError("No output file generated")
        
        preds = pd.read_csv(tmp_out)
        if preds.empty:
            return pd.DataFrame({id_col: test_df[id_col]})
        
        pred_col = [c for c in preds.columns if c != "smiles"]
        return pd.DataFrame({
            id_col: test_df[id_col].values,
            "y_chemprop": preds[pred_col[0]].values
        })
    except Exception as e:
        print(f"[WARN] Chemprop prediction failed: {e}")
        return pd.DataFrame({id_col: test_df[id_col]})


def predict_gbm(models_dir, test_df, id_col):
    """
    Generate predictions using GBM models (XGBoost, LightGBM, CatBoost).
    
    Args:
        models_dir: Directory containing model pickles
        test_df: Test DataFrame with features
        id_col: ID column name
    
    Returns:
        DataFrame with id and prediction columns
    """
    if test_df.empty:
        return pd.DataFrame(columns=[id_col])
    
    preds = pd.DataFrame({id_col: test_df[id_col]})
    
    # Load feature list from manifest
    features_gbm = []
    manifest = Path(models_dir) / "gbm_manifest.json"
    if manifest.exists():
        try:
            features_gbm = json.loads(manifest.read_text()).get("features", [])
        except Exception:
            pass
    
    def get_features(df, feats):
        """Extract feature matrix."""
        if feats:
            # Ensure all expected features exist
            for c in feats:
                if c not in df.columns:
                    df[c] = 0
            return df[feats]
        # Fallback: use all numeric columns
        drop_cols = [id_col, "fold", "target", "logS"]
        return df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')
    
    # XGBoost
    xgb_path = Path(models_dir) / "xgb.pkl"
    if xgb_path.exists():
        try:
            print("[Infer] XGBoost...")
            pipe = joblib.load(xgb_path)
            preds["y_xgb"] = pipe.predict(get_features(test_df.copy(), features_gbm))
        except Exception as e:
            print(f"[WARN] XGBoost prediction failed: {e}")
    
    # LightGBM
    lgbm_path = Path(models_dir) / "lgbm.pkl"
    if lgbm_path.exists():
        try:
            print("[Infer] LightGBM...")
            pipe = joblib.load(lgbm_path)
            preds["y_lgbm"] = pipe.predict(get_features(test_df.copy(), features_gbm))
        except Exception as e:
            print(f"[WARN] LightGBM prediction failed: {e}")
    
    # CatBoost
    cat_path = Path(models_dir) / "cat.pkl"
    if cat_path.exists():
        try:
            print("[Infer] CatBoost...")
            pipe = joblib.load(cat_path)
            preds["y_cat"] = pipe.predict(get_features(test_df.copy(), features_gbm))
        except Exception as e:
            print(f"[WARN] CatBoost prediction failed: {e}")
    
    return preds


def predict_physics(physics_json, test_df, id_col):
    """
    Generate predictions using physics-informed Ridge model.
    
    Args:
        physics_json: Path to physics model JSON file
        test_df: Test DataFrame with features
        id_col: ID column name
    
    Returns:
        DataFrame with id and y_physics columns
    """
    if test_df.empty or not physics_json or not Path(physics_json).exists():
        return pd.DataFrame(columns=[id_col])
    
    print("[Infer] Physics baseline...")
    
    try:
        model = json.loads(Path(physics_json).read_text())
        
        # Ensure inv_temp feature exists
        if "inv_temp" not in test_df.columns and "temp_C" in test_df.columns:
            t_celsius = pd.to_numeric(test_df["temp_C"], errors='coerce').fillna(25.0)
            test_df["inv_temp"] = 1000.0 / (t_celsius + 273.15)
        
        # Calculate predictions
        y_pred = np.full(len(test_df), model.get("intercept", 0.0))
        
        # Support both old (coefs) and new (coefficients) format
        coefs = model.get("coefficients", model.get("coefs", {}))
        
        for feat, weight in coefs.items():
            if feat in test_df.columns:
                values = pd.to_numeric(test_df[feat], errors='coerce').fillna(0.0).values
                y_pred += values * weight
        
        return pd.DataFrame({id_col: test_df[id_col], "y_physics": y_pred})
    
    except Exception as e:
        print(f"[WARN] Physics prediction failed: {e}")
        return pd.DataFrame({id_col: test_df[id_col]})


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for inference master."""
    
    ap = argparse.ArgumentParser(
        description="Generate ensemble predictions from all trained models."
    )
    
    # Input data
    ap.add_argument("--test-tabular", required=True,
                    help="Test data with tabular features")
    ap.add_argument("--test-smiles", required=True,
                    help="Test data with SMILES")
    
    # Model paths
    ap.add_argument("--models-dir", required=True,
                    help="Directory with GBM models")
    ap.add_argument("--chemprop-model-dir", required=True,
                    help="Directory with Chemprop model")
    ap.add_argument("--physics-json", default=None,
                    help="Physics model JSON file")
    
    # Ensemble weights
    ap.add_argument("--weights-json", default=None,
                    help="Blending weights JSON")
    ap.add_argument("--stack-pkl", default=None,
                    help="Stacking meta-model pickle")
    
    # Column names
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--target", default="logS")
    
    # Output
    ap.add_argument("--save-dir", default="pred")
    
    args = ap.parse_args()
    
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load Input Data
    # -------------------------------------------------------------------------
    print("[Infer] Loading inputs...")
    df_tab = read_any(args.test_tabular).drop_duplicates(subset=[args.id_col])
    df_smi = read_any(args.test_smiles).drop_duplicates(subset=[args.id_col])
    
    # Normalize IDs
    if not df_tab.empty:
        df_tab[args.id_col] = normalize_id(df_tab[args.id_col])
    if not df_smi.empty:
        df_smi[args.id_col] = normalize_id(df_smi[args.id_col])
    
    # Merge tabular and SMILES data
    df_full = df_tab.merge(df_smi, on=args.id_col, suffixes=("", "_smi"))
    
    if df_full.empty:
        print("[WARN] Empty dataset after merge")
        return
    
    print(f"[Infer] Loaded {len(df_full):,} samples")
    
    # Ensure inv_temp feature
    if "temp_C" in df_full.columns:
        t = pd.to_numeric(df_full["temp_C"], errors='coerce').fillna(25.0)
        df_full["inv_temp"] = 1000.0 / (t + 273.15)
    else:
        df_full["inv_temp"] = 1000.0 / 298.15  # Default 25°C
    
    # -------------------------------------------------------------------------
    # Generate Level-0 Predictions
    # -------------------------------------------------------------------------
    p_gbm = predict_gbm(args.models_dir, df_full, args.id_col)
    p_gnn = predict_chemprop(args.chemprop_model_dir, df_full, args.smiles_col, args.id_col, gpu=True)
    p_physics = predict_physics(args.physics_json, df_full, args.id_col)
    
    # Merge all predictions
    level0 = df_full[[args.id_col]].copy()
    level0 = level0.merge(p_gbm, on=args.id_col, how="left")
    level0 = level0.merge(p_gnn, on=args.id_col, how="left")
    level0 = level0.merge(p_physics, on=args.id_col, how="left")
    
    # Add metadata columns
    for col in [args.target, "temp_C"]:
        if col in df_full.columns:
            level0 = level0.merge(df_full[[args.id_col, col]], on=args.id_col, how="left")
    
    level0.to_parquet(outdir / "test_level0.parquet", index=False)
    print(f"[Infer] Level-0 predictions saved")
    
    # -------------------------------------------------------------------------
    # Generate Ensemble Predictions
    # -------------------------------------------------------------------------
    # Expected prediction columns
    pred_cols = ["y_xgb", "y_lgbm", "y_cat", "y_chemprop", "y_physics"]
    present_cols = [c for c in pred_cols if c in level0.columns]
    X = level0[present_cols].values
    
    # Blending
    if args.weights_json and Path(args.weights_json).exists():
        try:
            weights = json.loads(Path(args.weights_json).read_text())
            
            # Build weight vector matching present columns
            vec = np.array([weights.get(c.replace("y_", ""), 0.0) for c in present_cols])
            if vec.sum() > 0:
                vec /= vec.sum()  # Normalize
            
            level0["y_pred_blend"] = np.nansum(X * vec, axis=1)
            level0[[args.id_col, "y_pred_blend"]].to_parquet(outdir / "test_blend.parquet", index=False)
            print(f"[Infer] Blend predictions saved")
        except Exception as e:
            print(f"[WARN] Blending failed: {e}")
    
    # Stacking
    if args.stack_pkl and Path(args.stack_pkl).exists():
        try:
            # Try pickle first, then joblib
            try:
                meta = pickle.loads(Path(args.stack_pkl).read_bytes())
            except Exception:
                meta = joblib.load(args.stack_pkl)
            
            if "coef" in meta:
                y_stack = np.full(len(level0), meta["intercept"])
                
                for feat, weight in zip(meta["feature_names"], meta["coef"]):
                    if feat in level0.columns:
                        y_stack += level0[feat].fillna(0.0).values * weight
                
                level0["y_pred_stack"] = y_stack
                level0[[args.id_col, "y_pred_stack"]].to_parquet(outdir / "test_stack.parquet", index=False)
                print(f"[Infer] Stack predictions saved")
        except Exception as e:
            print(f"[WARN] Stacking failed: {e}")
    
    # -------------------------------------------------------------------------
    # Calculate Metrics
    # -------------------------------------------------------------------------
    if args.target in level0.columns:
        metrics = {}
        y_true = level0[args.target].values
        
        # Metrics for all models
        for col in present_cols + ["y_pred_blend", "y_pred_stack"]:
            if col in level0.columns:
                metrics[col] = calc_metrics(y_true, level0[col].values)
        
        # Physiological range metrics (35-38°C)
        if "temp_C" in level0.columns:
            t_val = pd.to_numeric(level0["temp_C"], errors='coerce')
            phys_mask = t_val.between(35, 38).values
            
            if phys_mask.any():
                metrics["physio_range"] = {}
                for col in present_cols + ["y_pred_blend", "y_pred_stack"]:
                    if col in level0.columns:
                        metrics["physio_range"][col] = calc_metrics(
                            y_true[phys_mask],
                            level0.loc[phys_mask, col].values
                        )
        
        # Save metrics
        with open(outdir / "metrics_test.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n[Infer] Metrics saved to metrics_test.json")


if __name__ == "__main__":
    main()
