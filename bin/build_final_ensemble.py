#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Final Ensemble: Production Model Packaging
=================================================

Packages the final production model by combining trained base models and
training the meta-learner on OOF predictions. Creates a deployable artifact
containing all components needed for inference.

Final Product Structure:
    drugsol_model/
    ├── base_models/
    │   ├── gbm/          # XGBoost, LightGBM, CatBoost models
    │   ├── gnn/          # Chemprop D-MPNN checkpoint
    │   └── physics/      # Physics-informed Ridge model
    ├── weights.json      # Blending weights (if strategy=blend)
    ├── meta_ridge.pkl    # Stacking meta-model (if strategy=stack)
    └── model_card.json   # Model metadata and version info

Meta-Learning Strategies:
    Blend:
        - Non-negative least squares (NNLS) to find optimal weights
        - Weights normalized to sum to 1
        - Simple, interpretable combination
    
    Stack:
        - Ridge regression meta-model
        - Cross-validated alpha selection
        - Can learn complex relationships

Arguments:
    --strategy      : Ensemble strategy (blend or stack)
    --oof-files     : OOF prediction parquet files
    --train-file    : Training data with ground truth
    --gbm-dir       : Directory with GBM models
    --gnn-dir       : Directory with GNN model
    --physics-model : Physics model file or directory
    --save-dir      : Output directory

Usage:
    python build_final_ensemble.py \\
        --strategy blend \\
        --oof-files oof_*.parquet \\
        --train-file train_data.parquet \\
        --gbm-dir models_GBM \\
        --gnn-dir models_GNN \\
        --physics-model models_physics \\
        --save-dir drugsol_model

Output:
    - Complete deployable model package in save-dir
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV


# ============================================================================
# UTILITIES
# ============================================================================

def copy_model_source(src: str, dest_folder: str):
    """
    Copy a model (file or directory) to destination.
    
    Args:
        src: Source path (file or directory)
        dest_folder: Destination directory
    """
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    src_path = Path(src)
    
    if not src_path.exists():
        print(f"[WARN] Model source not found: {src}")
        return
    
    if src_path.is_dir():
        print(f"[Build] Copying directory {src_path.name} to {dest_folder}...")
        shutil.copytree(src_path, dest_folder, dirs_exist_ok=True)
    else:
        print(f"[Build] Copying file {src_path.name} to {dest_folder}...")
        shutil.copy(src_path, dest_folder / src_path.name)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for building final ensemble."""
    
    ap = argparse.ArgumentParser(
        description="Build final production ensemble model."
    )
    ap.add_argument("--strategy", required=True,
                    choices=["blend", "stack"],
                    help="Ensemble strategy")
    ap.add_argument("--oof-files", nargs='+', required=True,
                    help="OOF prediction parquet files")
    ap.add_argument("--train-file", required=True,
                    help="Training data with ground truth")
    ap.add_argument("--gbm-dir", required=True,
                    help="GBM models directory")
    ap.add_argument("--gnn-dir", required=True,
                    help="GNN models directory")
    ap.add_argument("--physics-model", required=True,
                    help="Physics model file or directory")
    ap.add_argument("--save-dir", default="final_product",
                    help="Output directory")
    
    args = ap.parse_args()
    
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Build] Building Final Product with strategy: {args.strategy.upper()}")
    
    # -------------------------------------------------------------------------
    # Copy Base Models
    # -------------------------------------------------------------------------
    print("\n[Build] Packaging base models...")
    
    copy_model_source(args.gbm_dir, outdir / "base_models/gbm")
    copy_model_source(args.gnn_dir, outdir / "base_models/gnn")
    copy_model_source(args.physics_model, outdir / "base_models/physics")
    
    # -------------------------------------------------------------------------
    # Train Meta-Learner
    # -------------------------------------------------------------------------
    print(f"\n[Build] Loading {len(args.oof_files)} OOF files...")
    
    # Load OOF predictions
    dfs = []
    for f in args.oof_files:
        try:
            df_chunk = pd.read_parquet(f)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"[WARN] Error reading {f}: {e}")
    
    if not dfs:
        raise ValueError("No OOF files could be loaded")
    
    df_oof = pd.concat(dfs, ignore_index=True)
    
    # Load ground truth
    print(f"[Build] Loading ground truth from {args.train_file}...")
    df_train = pd.read_parquet(args.train_file)
    
    # Normalize ID column
    if 'id' not in df_train.columns and 'row_uid' in df_train.columns:
        df_train = df_train.rename(columns={'row_uid': 'id'})
    
    # Merge OOF with ground truth
    if 'id' in df_train.columns and 'logS' in df_train.columns:
        df = pd.merge(df_oof, df_train[['id', 'logS']], on='id', how='inner')
    else:
        print("[WARN] Could not merge by ID. Assuming same order (DANGEROUS).")
        if len(df_oof) == len(df_train):
            df = df_oof.copy()
            df['logS'] = df_train['logS'].values
        else:
            raise ValueError("ID mismatch and length mismatch. Cannot train meta-learner.")
    
    print(f"[Build] Meta-training dataset: {len(df):,} samples")
    
    # Detect prediction columns (updated: support physics naming)
    pred_cols = [c for c in df.columns if c.startswith("oof_") or c.startswith("y_pred_")]
    pred_cols = [c for c in pred_cols if "blend" not in c and "stack" not in c]
    pred_cols = sorted(pred_cols)
    
    print(f"[Build] Features detected: {pred_cols}")
    
    X = df[pred_cols].values
    y = df['logS'].values
    
    # Clean NaNs
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    print(f"[Build] Clean samples for meta-learning: {len(X):,}")
    
    # -------------------------------------------------------------------------
    # Train Strategy-Specific Meta-Model
    # -------------------------------------------------------------------------
    if args.strategy == "blend":
        print("\n[Build] Optimizing blend weights (NNLS)...")
        from scipy.optimize import nnls
        
        weights, _ = nnls(X, y)
        if weights.sum() > 0:
            weights /= weights.sum()
        
        # Standardize column names: oof_xgb -> y_xgb, y_pred_blend -> y_blend
        # Also update: oof_tpsa/oof_physics -> y_physics
        def standardize_name(col):
            name = col.replace("oof_", "y_").replace("y_pred_", "y_")
            # Handle legacy naming
            if name == "y_tpsa":
                name = "y_physics"
            return name
        
        w_dict = {standardize_name(col): float(w) for col, w in zip(pred_cols, weights)}
        
        with open(outdir / "weights.json", "w") as f:
            json.dump(w_dict, f, indent=2)
        
        print(f"[Build] Weights saved: {w_dict}")
    
    elif args.strategy == "stack":
        print("\n[Build] Training Stacker (RidgeCV)...")
        
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(X, y)
        
        # Standardize column names
        def standardize_name(col):
            name = col.replace("oof_", "y_").replace("y_pred_", "y_")
            if name == "y_tpsa":
                name = "y_physics"
            return name
        
        clean_names = [standardize_name(c) for c in pred_cols]
        
        stack_obj = {
            "model_type": "ridge",
            "feature_names": clean_names,
            "coef": list(model.coef_),
            "intercept": float(model.intercept_),
            "alpha": float(model.alpha_)
        }
        
        with open(outdir / "meta_ridge.pkl", "wb") as f:
            pickle.dump(stack_obj, f)
        
        print(f"[Build] Stacker saved. Alpha: {model.alpha_}")
        print(f"[Build] Coefficients: {dict(zip(clean_names, model.coef_.round(4)))}")
    
    # -------------------------------------------------------------------------
    # Save Model Card
    # -------------------------------------------------------------------------
    model_card = {
        "name": "DrugSol Aqueous Solubility Predictor",
        "version": "1.0.0",
        "strategy": args.strategy,
        "base_models": ["xgboost", "lightgbm", "catboost", "chemprop", "physics"],
        "target": "logS (log10 mol/L)",
        "description": "Ensemble model for predicting aqueous solubility of drug-like compounds"
    }
    
    with open(outdir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    
    print(f"\n[Build] Final product generated successfully in {outdir}/")


if __name__ == "__main__":
    main()
