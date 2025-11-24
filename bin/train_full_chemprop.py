#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_chemprop.py
----------------------
Retrains Chemprop on the full dataset using best HPs.
Features:
- Physics-Aware: Injects 1/T feature and Gaussian Weights (centered at 37C).
- GPU Acceleration.
- Compatible with Chemprop v2 API.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def run_cmd(cmd, check=True):
    print(f"[CMD] {' '.join(map(str, cmd))}")
    try:
        subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        raise e

def _accel_flags(use_gpu: bool):
    if use_gpu:
        return ["--accelerator", "gpu", "--devices", "1"]
    return []

def load_best_params(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

def main():
    ap = argparse.ArgumentParser(description="Full Train Chemprop")
    ap.add_argument("--train", required=True)
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="models/chemprop")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--best-params", required=True, help="JSON with best HPs")
    
    # Overrides
    ap.add_argument("--val-fraction", type=float, default=0.05, help="Internal validation split")
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"[Chemprop] Loading {args.train}...")
    df = pd.read_parquet(args.train) if args.train.endswith('.parquet') else pd.read_csv(args.train)
    
    # 2. Physics Engineering (Thermodynamics & Weights)
    if "temp_C" in df.columns:
        # 1/T (Kelvin)
        temps = df["temp_C"].fillna(25.0)
        df["inv_temp"] = 1000.0 / (temps + 273.15)
        
        # Gaussian Weights (Focus on 37C)
        # w = 1 + 2*exp(-(T-37)^2 / 2*8^2)
        sigma = 8.0
        df["sw_temp37"] = 1.0 + 2.0 * np.exp(-((temps - 37.0) ** 2) / (2 * sigma**2))
        print("[Chemprop] Applied Gaussian Weights (center=37C).")
    else:
        print("[WARN] No temp_C column. Using flat weights.")
        df["sw_temp37"] = 1.0
        # Can't calculate inv_temp without T

    # Clean rows
    cols = [args.smiles_col, args.target]
    if "inv_temp" in df.columns: cols.append("inv_temp")
    df = df.dropna(subset=cols).reset_index(drop=True)

    # Identify Descriptors (RDKit features if present)
    potential_descs = ["temp_C", "inv_temp", "n_ionizable", "n_acid", "n_base",
                       "TPSA", "logP", "HBD", "HBA", "FractionCSP3", "MW"]
    desc_cols = [c for c in potential_descs if c in df.columns]
    
    if desc_cols:
        print(f"[Chemprop] Using descriptors: {desc_cols}")

    # 3. Load Hyperparameters
    hp = load_best_params(args.best_params)
    
    epochs      = hp.get("epochs", args.epochs)
    hidden_size = int(hp.get("hidden_size", 1600))
    depth       = int(hp.get("depth", 3))
    dropout     = float(hp.get("dropout", 0.1))
    ffn_layers  = int(hp.get("ffn_num_layers", 1))
    batch_size  = int(hp.get("batch_size", 32))

    # 4. Prepare Training Files
    # Chemprop requires a CSV and a splits file
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_csv = tmp / "data.csv"
        splits_json = tmp / "splits.json"
        
        # Internal Split (Train / Val)
        # Even for "Full" training, we need a small validation set for Early Stopping
        n = len(df)
        n_val = max(1, int(n * args.val_fraction))
        n_train = n - n_val
        
        # Shuffle before split
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        
        # Save CSV
        df.to_csv(data_csv, index=False)
        
        splits = [{
            "train": list(range(n_train)),
            "val": list(range(n_train, n)),
            "test": []
        }]
        splits_json.write_text(json.dumps(splits))
        
        chemprop_bin = "chemprop" # Assumes in PATH
        
        cmd = [
            chemprop_bin, "train",
            "--data-path", str(data_csv),
            "--splits-file", str(splits_json),
            "--task-type", "regression",
            "--output-dir", str(outdir),
            "--save-dir", str(outdir),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--smiles-columns", args.smiles_col,
            "--target-columns", args.target,
            "--weights-column", "sw_temp37", # Modern Flag
            "--metric", "rmse",
            "--hidden-size", str(hidden_size),
            "--depth", str(depth),
            "--dropout", str(dropout),
            "--ffn-num-layers", str(ffn_layers),
            *(_accel_flags(args.gpu))
        ]
        
        if desc_cols:
            cmd += ["--descriptors-columns", *desc_cols]
            
        # Run Training
        run_cmd(cmd)
        
        # 5. Validate (Internal sanity check)
        # Predict on the validation set we held out
        val_df = df.iloc[n_train:].copy()
        val_in = tmp / "val_in.csv"
        val_out = tmp / "val_out.csv"
        val_df.to_csv(val_in, index=False)
        
        best_pt = next(outdir.glob("*.pt")) # Pick any checkpoint (usually best.pt is saved)
        
        cmd_pred = [
            chemprop_bin, "predict",
            "--test-path", str(val_in),
            "--preds-path", str(val_out),
            "--checkpoint-path", str(best_pt),
            "--batch-size", str(batch_size),
            "--drop-extra-columns",
            *(_accel_flags(args.gpu))
        ]
        if desc_cols:
            cmd_pred += ["--descriptors-columns", *desc_cols]
            
        run_cmd(cmd_pred)
        
        # Metrics
        preds = pd.read_csv(val_out)
        # Find pred column (not smiles)
        pred_col = [c for c in preds.columns if c != args.smiles_col and "smiles" not in c.lower()][0]
        
        y_true = val_df[args.target].values
        y_pred = preds[pred_col].values
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"[Full Chemprop] Validation RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
    # Save Manifest
    manifest = {
        "target": args.target,
        "smiles_col": args.smiles_col,
        "descriptors": desc_cols,
        "physics_aware": True,
        "weights_used": True,
        "n_samples": n
    }
    (outdir / "chemprop_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[Full Chemprop] Saved model to {outdir}")

if __name__ == "__main__":
    main()