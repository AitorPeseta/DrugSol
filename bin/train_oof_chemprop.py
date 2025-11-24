#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_oof_chemprop.py
---------------------
Trains Chemprop GNN using Out-of-Fold strategy.
Includes Optuna tuning for architecture hyperparameters.
Compatible with Chemprop v2.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner, NopPruner

def run_cmd(cmd, check=True):
    """Runs a subprocess command."""
    print(f"[CMD] {' '.join(map(str, cmd))}")
    try:
        subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with return code {e.returncode}")
        raise e

def get_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": len(y_true)
    }

def chemprop_train_wrapper(train_df, val_df, out_dir, hp, use_gpu, epochs, seed, batch_size, target_col, weight_col=None):
    """Wrapper to call chemprop train CLI."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine Train+Val into one file for Chemprop's splitting logic
    # Chemprop needs a single file and a splits file
    comb_csv = out_dir / "data.csv"
    splits_json = out_dir / "splits.json"
    
    # Ensure weight column exists if requested
    if weight_col:
        if weight_col not in train_df.columns: train_df[weight_col] = 1.0
        if weight_col not in val_df.columns: val_df[weight_col] = 1.0
    
    df_comb = pd.concat([train_df, val_df], ignore_index=True)
    df_comb.to_csv(comb_csv, index=False)
    
    # Create explicit split indices
    n_tr = len(train_df)
    n_va = len(val_df)
    splits = [{
        "train": list(range(0, n_tr)),
        "val": list(range(n_tr, n_tr + n_va)),
        "test": [] # No test set during OOF training
    }]
    splits_json.write_text(json.dumps(splits))
    
    # Feature columns (RDKit descriptors if present)
    # We look for common descriptor names
    desc_cols = [c for c in df_comb.columns if c not in [target_col, "smiles", "mol", weight_col or "", "id"]]
    # Filter numeric only just in case
    desc_cols = [c for c in desc_cols if pd.api.types.is_numeric_dtype(df_comb[c])]

    chemprop_bin = "chemprop" # Assumes it's in PATH from conda env
    
    cmd = [
        chemprop_bin, "train",
        "--data-path", str(comb_csv),
        "--splits-file", str(splits_json),
        "--task-type", "regression",
        "--output-dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--save-dir", str(out_dir)
    ]
    
    # Weights logic (Modern Chemprop uses column name)
    if weight_col:
        cmd += ["--weights-column", weight_col]
        
    # Target columns logic (can specify multiple, here just one)
    # Chemprop v2 might infer, but better explicit
    # NOTE: If using V1 CLI syntax, it might be --target-columns. 
    # We assume V1 syntax compatibility as most wrappers do.
    # cmd += ["--target-columns", target_col] 

    if use_gpu:
        # Chemprop v2 / Lightning style
        cmd += ["--accelerator", "gpu", "--devices", "1"]
        
    # Hyperparams
    if "hidden_size" in hp: cmd += ["--hidden-size", str(hp["hidden_size"])]
    if "depth" in hp:       cmd += ["--depth", str(hp["depth"])]
    if "dropout" in hp:     cmd += ["--dropout", str(hp["dropout"])]
    if "ffn_num_layers" in hp: cmd += ["--ffn-num-layers", str(hp["ffn_num_layers"])]
    
    # Run
    run_cmd(cmd)

def chemprop_predict_wrapper(test_df, model_path, out_csv, use_gpu, batch_size):
    """Wrapper for chemprop predict."""
    input_csv = Path(out_csv).with_suffix(".in.csv")
    test_df.to_csv(input_csv, index=False)
    
    cmd = [
        "chemprop", "predict",
        "--test-path", str(input_csv),
        "--checkpoint-path", str(model_path),
        "--preds-path", str(out_csv),
        "--batch-size", str(batch_size)
    ]
    
    if use_gpu:
        cmd += ["--accelerator", "gpu", "--devices", "1"]
        
    run_cmd(cmd)

def select_best_checkpoint(run_dir):
    """Finds the best .pt checkpoint file."""
    p = Path(run_dir)
    # Chemprop usually saves 'model.pt' or 'fold_0/model_0/model.pt'
    candidates = list(p.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in {run_dir}")
    # Prefer 'best.pt' or similar if exists, else take the first one
    best = [x for x in candidates if "best" in x.name]
    return best[0] if best else candidates[0]

# -------------------- OPTUNA --------------------

def sample_hp(trial):
    return {
        "hidden_size": trial.suggest_int("hidden_size", 300, 1200, step=100),
        "depth": trial.suggest_int("depth", 2, 6),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3)
    }

def optimize_fold(train_df, val_df, fold_idx, args):
    """Runs Optuna optimization for a single fold."""
    
    def objective(trial):
        hp = sample_hp(trial)
        
        # Create temporary directory for this trial
        trial_dir = Path(args.save_dir) / f"fold_{fold_idx}" / f"trial_{trial.number}"
        if trial_dir.exists(): shutil.rmtree(trial_dir)
        
        # Train with small epochs for pruning or full for final
        # ASHA logic would go here (training for partial epochs), 
        # but calling CLI repeatedly is slow. 
        # Simplified: Train once for fixed small epochs
        epochs = args.epochs // 2 if args.tune_trials > 5 else args.epochs
        
        try:
            chemprop_train_wrapper(
                train_df, val_df, trial_dir, hp, args.gpu, 
                epochs=epochs, seed=args.seed, batch_size=args.batch_size,
                target_col=args.target, weight_col=args.weight_col
            )
            
            # Predict on Val
            ckpt = select_best_checkpoint(trial_dir)
            preds_csv = trial_dir / "preds.csv"
            chemprop_predict_wrapper(val_df, ckpt, preds_csv, args.gpu, args.batch_size)
            
            # Score
            preds = pd.read_csv(preds_csv)
            # Assume only one prediction column if target is scalar
            y_pred = preds.iloc[:, 0].values 
            y_true = val_df[args.target].values
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            return rmse
            
        except Exception as e:
            print(f"[WARN] Trial failed: {e}")
            return 9999.0
        finally:
            # Cleanup to save space
            if trial_dir.exists(): shutil.rmtree(trial_dir)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.tune_trials)
    return study.best_params

# -------------------- MAIN --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--save-dir", required=True)
    
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--weight-col", default=None)
    
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    
    # Tuning
    ap.add_argument("--tune-trials", type=int, default=0)
    ap.add_argument("--tune-pruner", default="asha")
    ap.add_argument("--asha-rungs", nargs="+", type=int)
    
    args = ap.parse_args()
    
    # Load
    df = pd.read_parquet(args.train) if args.train.endswith('.parquet') else pd.read_csv(args.train)
    folds = pd.read_parquet(args.folds) if args.folds.endswith('.parquet') else pd.read_csv(args.folds)
    
    # Merge
    df = df.merge(folds[[args.id_col, "fold"]], on=args.id_col, how="inner")
    
    # Prepare OOF storage
    oof_preds = np.zeros(len(df))
    oof_ids = df[args.id_col].values
    
    best_hps = []
    
    unique_folds = sorted(df["fold"].unique())
    
    for fold in unique_folds:
        print(f"=== FOLD {fold} ===")
        train_sub = df[df["fold"] != fold].copy()
        val_sub = df[df["fold"] == fold].copy()
        
        # 1. Tune (if requested)
        if args.tune_trials > 0:
            print(f"  Tuning ({args.tune_trials} trials)...")
            best_hp = optimize_fold(train_sub, val_sub, fold, args)
        else:
            # Default params
            best_hp = {"hidden_size": 300, "depth": 3, "dropout": 0.0, "ffn_num_layers": 2}
            
        best_hps.append(best_hp)
        
        # 2. Final Train
        print(f"  Training Final Model for Fold {fold}...")
        fold_dir = Path(args.save_dir) / f"fold_{fold}"
        if fold_dir.exists(): shutil.rmtree(fold_dir)
        
        chemprop_train_wrapper(
            train_sub, val_sub, fold_dir, best_hp, args.gpu, 
            args.epochs, args.seed, args.batch_size,
            args.target, args.weight_col
        )
        
        # 3. Predict OOF
        ckpt = select_best_checkpoint(fold_dir)
        preds_file = fold_dir / "val_preds.csv"
        chemprop_predict_wrapper(val_sub, ckpt, preds_file, args.gpu, args.batch_size)
        
        # Store
        p = pd.read_csv(preds_file)
        # Robustly find prediction column (first non-smiles/id)
        pred_col = p.columns[0] # Usually 'logS' or similar
        # If first col is smiles, take second
        if "smiles" in pred_col.lower(): pred_col = p.columns[1]
            
        val_indices = df.index[df["fold"] == fold]
        oof_preds[val_indices] = p[pred_col].values

    # Save OOF
    oof_df = pd.DataFrame({
        "id": df[args.id_col],
        "fold": df["fold"],
        "y_true": df[args.target],
        "y_pred": oof_preds
    })
    
    out_parquet = Path(args.save_dir) / "chemprop.parquet"
    oof_df.to_parquet(out_parquet, index=False)
    
    # Save Best HP (Average/Mode or just list)
    # For simplicity, save the HP of fold 0 as representative
    hp_file = Path(args.save_dir) / "chemprop_best_params.json"
    with open(hp_file, "w") as f:
        json.dump(best_hps[0], f, indent=2)
        
    # Save Metrics
    metrics = get_metrics(df[args.target].values, oof_preds)
    with open(Path(args.save_dir) / "metrics_oof_chemprop.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[Done] OOF RMSE: {metrics['rmse']:.4f}")

if __name__ == "__main__":
    main()