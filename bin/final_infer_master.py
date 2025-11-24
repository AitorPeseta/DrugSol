#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_infer_master.py
---------------------
Orchestrates the final inference process on the Test set.
1. Generates predictions from base models (XGB, LGBM, Chemprop, TPSA).
2. Applies Stacking/Blending meta-models.
3. Calculates performance metrics (Global + Physiological Range).
"""

import argparse
import json
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- UTILS ----------------

def run_cmd(cmd, check=True):
    print(f"[CMD] {' '.join(map(str, cmd))}")
    try:
        subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        raise e

def read_any(path):
    p = Path(path)
    if p.suffix == '.parquet': return pd.read_parquet(p)
    return pd.read_csv(p)

def get_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return {}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "r2": float(r2_score(y_true[mask], y_pred[mask])),
        "n": int(mask.sum())
    }

# ---------------- MODEL INFERENCE ----------------

def predict_gbm(models_dir, test_df, id_col):
    """Generates predictions for XGBoost and LightGBM."""
    preds = pd.DataFrame({id_col: test_df[id_col]})
    
    # XGBoost
    xgb_path = Path(models_dir) / "xgb.pkl"
    if xgb_path.exists():
        try:
            print("[Infer] Predicting with XGBoost...")
            xgb_pipe = joblib.load(xgb_path)
            # Ensure features match training
            # (Pipeline handles scaling/imputing, but we need correct columns)
            # We rely on the pipeline's feature names if available, or intersect
            
            # Get feature names from the VarianceThreshold step or similar if possible
            # Or just try to predict and let sklearn complain nicely
            
            # Drop non-features
            X = test_df.select_dtypes(include=[np.number])
            cols_to_drop = [id_col, "fold", "target", "logS"]
            X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
            
            # Align columns with training manifest if present
            manifest_path = Path(models_dir) / "gbm_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                feat_cols = manifest.get("features", [])
                # Add missing as 0, drop extra
                for c in feat_cols:
                    if c not in X.columns: X[c] = 0
                X = X[feat_cols] # Reorder
            
            preds["y_xgb"] = xgb_pipe.predict(X)
        except Exception as e:
            print(f"[WARN] XGBoost inference failed: {e}")

    # LightGBM
    lgbm_path = Path(models_dir) / "lgbm.pkl"
    if lgbm_path.exists():
        try:
            print("[Infer] Predicting with LightGBM...")
            lgbm_pipe = joblib.load(lgbm_path)
            # Re-use X from above if possible
            preds["y_lgbm"] = lgbm_pipe.predict(X)
        except Exception as e:
            print(f"[WARN] LightGBM inference failed: {e}")
            
    return preds

def predict_chemprop(model_dir, test_smiles_df, smiles_col, id_col, gpu=False):
    """Generates predictions for Chemprop."""
    # Locate best checkpoint
    p = Path(model_dir)
    # Prioritize: output/best.pt -> output/model_0/best.pt -> recursive search
    ckpt = p / "best.pt"
    if not ckpt.exists():
        cands = list(p.rglob("best.pt"))
        if cands: ckpt = cands[0]
        else:
            # Fallback to any .pt
            cands = list(p.rglob("*.pt"))
            if not cands:
                print("[WARN] No Chemprop checkpoint found.")
                return pd.DataFrame({id_col: test_smiles_df[id_col]})
            ckpt = cands[0]
            
    print(f"[Infer] Predicting with Chemprop (ckpt={ckpt.name})...")
    
    # Prepare Input CSV
    tmp_in = p / "test_in.csv"
    tmp_out = p / "test_out.csv"
    
    # Chemprop expects SMILES as first column
    cols = [smiles_col] + [c for c in test_smiles_df.columns if c not in [smiles_col, id_col, "smiles"]]
    # Ensure descriptor columns are present if model needs them
    # For now, dump everything numeric + smiles
    df_in = test_smiles_df.rename(columns={smiles_col: "smiles"})
    # Keep numeric columns for descriptors
    desc_cols = ["temp_C", "inv_temp", "n_ionizable", "n_acid", "n_base", "TPSA", "logP", "HBD", "HBA", "FractionCSP3", "MW"]
    present_desc = [c for c in desc_cols if c in df_in.columns]
    
    df_in_final = df_in[["smiles"] + present_desc]
    df_in_final.to_csv(tmp_in, index=False)
    
    cmd = [
        "chemprop", "predict",
        "--test-path", str(tmp_in),
        "--preds-path", str(tmp_out),
        "--checkpoint-path", str(ckpt),
        "--batch-size", "64",
        "--drop-extra-columns"
    ]
    if gpu: cmd += ["--accelerator", "gpu", "--devices", "1"]
    if present_desc: cmd += ["--descriptors-columns", *present_desc]
    
    try:
        run_cmd(cmd)
        preds = pd.read_csv(tmp_out)
        
        # Find pred column
        # Usually 'logS' or 'prediction'
        target_col = preds.columns[0] if "smiles" not in preds.columns[0].lower() else preds.columns[1]
        # Robust check
        for c in preds.columns:
            if c.lower() not in ["smiles"] + [d.lower() for d in present_desc]:
                target_col = c
                break
        
        out = pd.DataFrame({
            id_col: test_smiles_df[id_col].values,
            "y_chemprop": preds[target_col].values
        })
        return out
    except Exception as e:
        print(f"[WARN] Chemprop inference failed: {e}")
        return pd.DataFrame({id_col: test_smiles_df[id_col]})

def predict_tpsa(tpsa_json_path, test_df, id_col):
    """Generates predictions for Baseline TPSA/Ridge."""
    if not tpsa_json_path or not Path(tpsa_json_path).exists():
        return pd.DataFrame({id_col: test_df[id_col]})
        
    print("[Infer] Predicting with Baseline (Ridge)...")
    try:
        model = json.loads(Path(tpsa_json_path).read_text())
        
        # Calculate features on the fly if missing
        if "inv_temp" not in test_df.columns and "temp_C" in test_df.columns:
             test_df["inv_temp"] = 1000.0 / (test_df["temp_C"] + 273.15)
             
        # Formula: y = intercept + sum(coef * val)
        intercept = model.get("intercept", 0.0)
        coefs = model.get("coefs", {}) # {feature_name: weight}
        
        y_pred = np.full(len(test_df), intercept)
        
        for feat, w in coefs.items():
            if feat in test_df.columns:
                vals = pd.to_numeric(test_df[feat], errors='coerce').fillna(0.0).values
                y_pred += vals * w
            else:
                print(f"[WARN] TPSA feature missing: {feat}")
                
        return pd.DataFrame({id_col: test_df[id_col], "y_tpsa": y_pred})
        
    except Exception as e:
        print(f"[WARN] TPSA inference failed: {e}")
        return pd.DataFrame({id_col: test_df[id_col]})

# ---------------- MAIN ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-tabular", required=True)
    ap.add_argument("--test-smiles", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--chemprop-model-dir", required=True)
    ap.add_argument("--tpsa-json", default=None)
    ap.add_argument("--save-dir", default="pred")
    
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--target", default="logS")
    
    # Ignored but kept for compatibility
    ap.add_argument("--chemprop-smiles-col", default=None)
    ap.add_argument("--tpsa-col", default="TPSA")
    ap.add_argument("--phenol-col", default="n_phenol")
    
    ap.add_argument("--weights-json", default=None)
    ap.add_argument("--stack-pkl", default=None)
    
    args = ap.parse_args()
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load Test Data
    print("[Infer] Loading Test Data...")
    df_tab = read_any(args.test_tabular)
    df_smi = read_any(args.test_smiles)
    
    # Ensure ID alignment
    if args.id_col not in df_tab.columns or args.id_col not in df_smi.columns:
        sys.exit(f"[ERROR] ID col '{args.id_col}' missing.")
        
    # Merge to have all features available
    # (Tabular has descriptors, SMILES has... smiles)
    # We perform a left merge on Tabular to keep its rows
    df_full = df_tab.merge(df_smi, on=args.id_col, suffixes=("", "_smi"))
    
    # 2. Base Predictions
    preds_gbm = predict_gbm(args.models_dir, df_full, args.id_col)
    preds_gnn = predict_chemprop(args.chemprop_model_dir, df_full, args.smiles_col, args.id_col)
    preds_tpsa = predict_tpsa(args.tpsa_json, df_full, args.id_col)
    
    # Combine Level 0
    level0 = preds_gbm.merge(preds_gnn, on=args.id_col, how="left") \
                      .merge(preds_tpsa, on=args.id_col, how="left")
                      
    # Add Target if available (for metrics)
    if args.target in df_full.columns:
        level0 = level0.merge(df_full[[args.id_col, args.target]], on=args.id_col, how="left")
        
    # Save Level 0
    level0.to_parquet(outdir / "test_level0.parquet", index=False)
    
    # 3. Ensemble
    cols = ["y_xgb", "y_lgbm", "y_chemprop", "y_tpsa"]
    present_cols = [c for c in cols if c in level0.columns]
    X_meta = level0[present_cols].values
    
    # A. Blending
    if args.weights_json and Path(args.weights_json).exists():
        print("[Infer] Applying Blend Weights...")
        weights = json.loads(Path(args.weights_json).read_text())
        # Align weights to columns
        w_vec = np.array([weights.get(c.replace("y_", ""), 0.0) for c in present_cols])
        # Normalize
        if w_vec.sum() > 0: w_vec /= w_vec.sum()
        
        y_blend = np.nansum(X_meta * w_vec, axis=1)
        level0["y_pred_blend"] = y_blend
        level0[[args.id_col, "y_pred_blend"]].to_parquet(outdir / "test_blend.parquet", index=False)

    # B. Stacking
    if args.stack_pkl and Path(args.stack_pkl).exists():
        print("[Infer] Applying Stacking Model...")
        stack_meta = pickle.loads(Path(args.stack_pkl).read_bytes())
        # Check model type
        if "coef" in stack_meta:
            # It's our custom dictionary from meta_stack_blend.py
            coefs = stack_meta["coef"]
            intercept = stack_meta["intercept"]
            # Map features
            feat_names = stack_meta["feature_names"] # e.g. ["y_xgb", "y_lgbm"]
            
            y_stack = np.full(len(level0), intercept)
            for f, w in zip(feat_names, coefs):
                if f in level0.columns:
                    y_stack += level0[f].fillna(0.0).values * w
                    
            level0["y_pred_stack"] = y_stack
            level0[[args.id_col, "y_pred_stack"]].to_parquet(outdir / "test_stack.parquet", index=False)

    # 4. Metrics (if Target exists)
    if args.target in level0.columns:
        print("[Infer] Calculating Metrics...")
        metrics = {}
        y_true = level0[args.target].values
        
        # Global Metrics
        for c in present_cols + ["y_pred_blend", "y_pred_stack"]:
            if c in level0.columns:
                metrics[c] = get_metrics(y_true, level0[c].values)
                
        # Physiological Range Metrics (35-38 C)
        if "temp_C" in df_full.columns:
            phys_mask = df_full["temp_C"].between(35, 38).values
            if phys_mask.any():
                metrics["physio_range"] = {}
                for c in present_cols + ["y_pred_blend", "y_pred_stack"]:
                    if c in level0.columns:
                        metrics["physio_range"][c] = get_metrics(y_true[phys_mask], level0.loc[phys_mask, c].values)
        
        (outdir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
        print(f"[Infer] Done. Blend RMSE: {metrics.get('y_pred_blend', {}).get('rmse', 'N/A')}")

if __name__ == "__main__":
    main()