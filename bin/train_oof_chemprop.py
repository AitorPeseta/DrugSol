#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train OOF Chemprop: Graph Neural Network Training with Out-of-Fold Predictions
===============================================================================

Trains a Chemprop D-MPNN (Directed Message Passing Neural Network) for molecular
property prediction using only SMILES structure. External descriptors are
intentionally excluded to let the model learn directly from molecular graphs.

Model Architecture:
    Chemprop uses directed message passing over molecular graphs:
    1. Parse SMILES to molecular graph (atoms as nodes, bonds as edges)
    2. Initialize atom/bond features from chemistry
    3. Iteratively pass messages along directed edges
    4. Aggregate node features to molecular representation
    5. Feed through FFN to produce final prediction

Training Strategy:
    - Pure graph approach: only SMILES structure, no Mordred/RDKit features
    - K-fold cross-validation with scaffold-aware splits
    - Optional Optuna hyperparameter tuning
    - Sample weighting for temperature-importance adjustment

Arguments:
    --train        : Training data file (Parquet/CSV)
    --folds        : Fold assignments file (Parquet/CSV)
    --smiles-col   : SMILES column name (default: smiles_neutral)
    --id-col       : Row identifier column (default: row_uid)
    --target       : Target column name (default: logS)
    --weight-col   : Sample weight column (default: None)
    --epochs       : Training epochs (default: 40)
    --batch-size   : Batch size (default: 32)
    --tune-trials  : Optuna trials (0 = no tuning)
    --tune-pruner  : Pruner type (default: asha)
    --asha-rungs   : ASHA checkpoint epochs (default: auto)
    --save-dir     : Output directory

Usage:
    python train_oof_chemprop.py \\
        --train train_data.parquet \\
        --folds folds.parquet \\
        --smiles-col smiles_neutral \\
        --target logS \\
        --epochs 50 \\
        --tune-trials 20 \\
        --save-dir ./oof_gnn

Output:
    - chemprop.parquet: OOF predictions
    - chemprop_best_params.json: Best hyperparameters
    - metrics_oof_chemprop.json: Performance metrics
    - fold_*/: Per-fold model checkpoints

Notes:
    - Requires chemprop package installed
    - GPU recommended for reasonable training times
    - Pure graph strategy avoids feature engineering noise
"""

import os
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

os.environ["TORCH_ALLOW_W_O_SERIALIZATION"] = "1"

import torch
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([argparse.Namespace])

try:
    import optuna
    from optuna.pruners import HyperbandPruner, NopPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ============================================================================
# I/O UTILITIES
# ============================================================================

def read_data(path: str) -> pd.DataFrame:
    """Read Parquet or CSV file."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported format: {p.suffix}")


def calc_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "r2": float(r2), "n": int(len(y_true))}


# ============================================================================
# CHEMPROP INTERFACE
# ============================================================================

def run_command(cmd, check=True):
    """Run shell command with logging and PyTorch security bypass."""
    print(f"[CMD] {' '.join(map(str, cmd))}")
    env = os.environ.copy()
    env["TORCH_ALLOW_W_O_SERIALIZATION"] = "1"
    subprocess.run(cmd, check=check, env=env)


def chemprop_train(
    train_csv, val_csv, out_dir, hp, use_gpu, epochs, seed,
    batch_size, target_col, weight_col=None, use_weights=True
):
    """
    Train Chemprop model using CLI interface.
    
    Args:
        train_csv: Training data CSV path
        val_csv: Validation data CSV path (optional)
        out_dir: Output directory for checkpoints
        hp: Hyperparameters dictionary
        use_gpu: Whether to use GPU
        epochs: Number of training epochs
        seed: Random seed
        batch_size: Training batch size
        target_col: Target column name
        weight_col: Sample weight column (optional)
        use_weights: Whether to apply sample weights
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    comb_csv = out_dir / "data.csv"
    weights_csv = out_dir / "weights.csv"
    val_dummy_csv = out_dir / "val_dummy.csv"
    
    df_tr = pd.read_csv(train_csv)
    
    # Pure graph strategy: only SMILES and target
    cols_to_save = ["smiles_neutral", target_col]
    apply_weights = (weight_col is not None and use_weights)
    
    # Build command
    cmd = [
        "chemprop_train",
        "--dataset_type", "regression",
        "--save_dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--smiles_columns", "smiles_neutral",
        "--target_columns", target_col,
        "--metric", "rmse",
    ]
    
    if apply_weights:
        # Strategy with weights (requires dummy validation)
        if weight_col not in df_tr.columns:
            df_tr[weight_col] = 1.0
        
        df_tr[cols_to_save].to_csv(comb_csv, index=False)
        df_tr[[weight_col]].to_csv(weights_csv, index=False, header=True)
        df_tr.iloc[:5][cols_to_save].to_csv(val_dummy_csv, index=False)
        
        cmd += [
            "--data_path", str(comb_csv),
            "--separate_val_path", str(val_dummy_csv),
            "--separate_test_path", str(val_dummy_csv),
            "--data_weights_path", str(weights_csv)
        ]
    else:
        # Strategy for Optuna (random split)
        if val_csv:
            df_va = pd.read_csv(val_csv)
            df_comb = pd.concat([df_tr, df_va], ignore_index=True)
            df_comb[cols_to_save].to_csv(comb_csv, index=False)
            
            n_tot = len(df_comb)
            n_tr = len(df_tr)
            ftr = n_tr / n_tot
            fva = 1.0 - ftr
            cmd += [
                "--data_path", str(comb_csv),
                "--split_type", "random",
                "--split_sizes", f"{ftr:.4f}", f"{fva:.4f}", "0.0"
            ]
        else:
            df_tr[cols_to_save].to_csv(comb_csv, index=False)
            cmd += [
                "--data_path", str(comb_csv),
                "--split_type", "random",
                "--split_sizes", "0.8", "0.2", "0.0"
            ]
    
    # GPU flag
    if use_gpu:
        cmd += ["--gpu", "0"]
    
    # Hyperparameters
    if "message_hidden_dim" in hp:
        cmd += ["--hidden_size", str(int(hp["message_hidden_dim"]))]
    if "depth" in hp:
        cmd += ["--depth", str(int(hp["depth"]))]
    if "dropout" in hp:
        cmd += ["--dropout", str(float(hp["dropout"]))]
    if "ffn_num_layers" in hp:
        cmd += ["--ffn_num_layers", str(int(hp["ffn_num_layers"]))]
    
    run_command(cmd, check=True)


def chemprop_predict(val_csv, model_path, out_csv, gpu, batch_size, smiles_col):
    """
    Generate predictions using trained Chemprop model.
    
    Args:
        val_csv: Input data CSV path
        model_path: Path to model checkpoint (.pt)
        out_csv: Output predictions CSV path
        gpu: Whether to use GPU
        batch_size: Prediction batch size
        smiles_col: SMILES column name
    """
    df = pd.read_csv(val_csv)
    tmp_in = Path(out_csv).parent / "pred_in.csv"
    
    # Only SMILES for prediction
    df[[smiles_col]].to_csv(tmp_in, index=False)
    
    cmd = [
        "chemprop_predict",
        "--test_path", str(tmp_in),
        "--preds_path", str(out_csv),
        "--checkpoint_path", str(model_path),
        "--batch_size", str(batch_size),
        "--smiles_columns", smiles_col
    ]
    
    if gpu:
        cmd += ["--gpu", "0"]
    
    run_command(cmd, check=True)


def select_best_checkpoint(run_dir: Path) -> Path:
    """Find best model checkpoint in directory."""
    run_dir = Path(run_dir)
    ckpts = sorted(list(run_dir.rglob("*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No .pt checkpoint found in {run_dir}")
    
    # Prefer model.pt (final checkpoint)
    best = [c for c in ckpts if c.name == "model.pt"]
    return best[0] if best else ckpts[0]


# ============================================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================================

def sample_hyperparameters(trial):
    """Sample Chemprop hyperparameters for Optuna trial."""
    return {
        "message_hidden_dim": trial.suggest_int("message_hidden_dim", 300, 800, step=100),
        "depth": trial.suggest_int("depth", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 2)
    }


def default_rungs(max_epochs):
    """Generate default ASHA checkpoint epochs."""
    return [int(max_epochs * 0.25), int(max_epochs * 0.5), max_epochs]


def get_pruner(name):
    """Get Optuna pruner by name."""
    if not HAS_OPTUNA:
        return None
    # For simplicity, return NopPruner (pruning handled manually via rungs)
    return NopPruner()


def write_fold_csv(df_tr, df_va, smiles_col, target, out_train, out_val):
    """Write fold data to CSV files."""
    cols = [smiles_col, target]
    df_tr[cols].to_csv(out_train, index=False)
    df_va[cols].to_csv(out_val, index=False)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for Chemprop OOF training."""
    
    ap = argparse.ArgumentParser(
        description="Train Chemprop D-MPNN with out-of-fold predictions."
    )
    
    # Data inputs
    ap.add_argument("--train", required=True, help="Training data file")
    ap.add_argument("--folds", required=True, help="Fold assignments file")
    ap.add_argument("--smiles-col", required=True, help="SMILES column name")
    ap.add_argument("--id-col", required=True, help="ID column name")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--weight-col", default=None, help="Sample weight column")
    
    # Output
    ap.add_argument("--save-dir", required=True, help="Output directory")
    
    # Training settings
    ap.add_argument("--epochs", type=int, default=40, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--gpu", action="store_true", help="Use GPU")
    
    # Optuna settings
    ap.add_argument("--tune-trials", type=int, default=0, help="Optuna trials (0=none)")
    ap.add_argument("--tune-timeout", type=int, default=None, help="Tuning timeout (seconds)")
    ap.add_argument("--tune-pruner", default="asha", help="Pruner type")
    ap.add_argument("--asha-rungs", nargs="*", type=int, default=None, help="ASHA checkpoint epochs")
    
    # Compatibility
    ap.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    
    args = ap.parse_args()
    
    # Setup output directory
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("[Chemprop] Loading data...")
    df = read_data(args.train).copy()
    folds = read_data(args.folds).copy()
    
    # Merge folds
    df = df.merge(folds[[args.id_col, "fold"]], on=args.id_col, how="inner")
    
    # Initialize weight column if missing
    if args.weight_col and args.weight_col not in df.columns:
        df[args.weight_col] = 1.0
    
    print(f"[Chemprop] Loaded {len(df):,} samples")
    
    # -------------------------------------------------------------------------
    # Initialize OOF DataFrame
    # -------------------------------------------------------------------------
    oof = pd.DataFrame({
        args.id_col: df[args.id_col],
        "fold": df["fold"],
        "y_true": df[args.target]
    })
    oof["y_hat"] = np.nan
    
    folds_unique = sorted(df["fold"].unique().tolist())
    rungs = sorted(args.asha_rungs) if args.asha_rungs else default_rungs(args.epochs)
    best_by_fold = {}
    
    # -------------------------------------------------------------------------
    # Train Each Fold
    # -------------------------------------------------------------------------
    for k in folds_unique:
        print(f"\n{'='*60}")
        print(f" FOLD {k}")
        print(f"{'='*60}")
        
        fold_dir = outdir / f"fold_{k}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Split data
        tr = df[df["fold"] != k].copy()
        va = df[df["fold"] == k].copy()
        
        print(f"[Fold {k}] Train: {len(tr):,}, Val: {len(va):,}")
        
        tr_csv = fold_dir / "train.csv"
        va_csv = fold_dir / "val.csv"
        write_fold_csv(tr, va, args.smiles_col, args.target, tr_csv, va_csv)
        
        # ---------------------------------------------------------------------
        # Hyperparameter Tuning (Optional)
        # ---------------------------------------------------------------------
        if args.tune_trials > 0 and HAS_OPTUNA:
            print(f"[Fold {k}] Running Optuna ({args.tune_trials} trials)...")
            
            study = optuna.create_study(
                direction="minimize",
                pruner=get_pruner(args.tune_pruner)
            )
            
            def objective(trial):
                hp = sample_hyperparameters(trial)
                
                # Inner train/val split
                from sklearn.model_selection import train_test_split
                tri, vai = train_test_split(tr, test_size=0.2, random_state=args.seed)
                
                tdir = fold_dir / f"trial_{trial.number}"
                if tdir.exists():
                    shutil.rmtree(tdir)
                tdir.mkdir(parents=True, exist_ok=True)
                
                tri_csv = tdir / "tri.csv"
                vai_csv = tdir / "vai.csv"
                write_fold_csv(tri, vai, args.smiles_col, args.target, tri_csv, vai_csv)
                
                best_rmse = None
                try:
                    for r_ep in rungs:
                        r_dir = tdir / f"e{r_ep}"
                        chemprop_train(
                            tri_csv, vai_csv, r_dir, hp, args.gpu, r_ep,
                            args.seed, args.batch_size, args.target,
                            args.weight_col, use_weights=False
                        )
                        
                        pt = select_best_checkpoint(r_dir)
                        p_out = r_dir / "pred.csv"
                        chemprop_predict(vai_csv, pt, p_out, args.gpu, args.batch_size, args.smiles_col)
                        
                        pred = pd.read_csv(p_out)
                        if args.target in pred.columns:
                            vals = pred[args.target].values
                        else:
                            vals = pred.iloc[:, 0].values
                        
                        rmse = np.sqrt(mean_squared_error(vai[args.target].values, vals))                               
                        best_rmse = rmse
                        
                        trial.report(rmse, r_ep)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                finally:
                    shutil.rmtree(tdir, ignore_errors=True)
                
                return best_rmse
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=args.tune_trials)
            
            print(f"[Fold {k}] Best params: {study.best_trial.params}")
            params = study.best_trial.params
            
            best_hp = {
                "message_hidden_dim": params.get("message_hidden_dim", 300),
                "depth": params.get("depth", 3),
                "dropout": params.get("dropout", 0.0),
                "ffn_num_layers": params.get("ffn_num_layers", 2)
            }
            best_by_fold[k] = best_hp
        else:
            # Default hyperparameters
            best_hp = {
                "message_hidden_dim": 300,
                "depth": 3,
                "dropout": 0.0,
                "ffn_num_layers": 2
            }
            best_by_fold[k] = best_hp
        
        # ---------------------------------------------------------------------
        # Final Training
        # ---------------------------------------------------------------------
        print(f"[Fold {k}] Training final model...")
        fin_dir = fold_dir / "final"
        chemprop_train(
            tr_csv, None, fin_dir, best_hp, args.gpu, args.epochs,
            args.seed, args.batch_size, args.target,
            args.weight_col, use_weights=True
        )
        
        # ---------------------------------------------------------------------
        # Generate OOF Predictions
        # ---------------------------------------------------------------------
        pt = select_best_checkpoint(fin_dir)
        oof_out = fold_dir / "oof_pred.csv"
        chemprop_predict(va_csv, pt, oof_out, args.gpu, args.batch_size, args.smiles_col)
        
        pred = pd.read_csv(oof_out)
        if args.target in pred.columns:
            vals = pred[args.target].values
        else:
            vals = pred.iloc[:, 0].values
        
        oof.loc[oof["fold"] == k, "y_hat"] = vals
        
        fold_rmse = np.sqrt(mean_squared_error(va[args.target].values, vals))        
        print(f"[Fold {k}] RMSE: {fold_rmse:.4f}")
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    # OOF predictions
    common = pd.DataFrame({
        "id": df[args.id_col],
        "fold": df["fold"],
        "y_true": df[args.target],
        "y_pred": oof["y_hat"]
    })
    common.to_parquet(outdir / "chemprop.parquet", index=False)
    
    # Best parameters (use first fold's params as representative)
    with open(outdir / "chemprop_best_params.json", "w") as f:
        json.dump(list(best_by_fold.values())[0], f, indent=2)
    
    # Metrics
    m = calc_metrics(common["y_true"], common["y_pred"])
    with open(outdir / "metrics_oof_chemprop.json", "w") as f:
        json.dump(m, f, indent=2)
    
    print(f"\n[Chemprop] OOF Results:")
    print(f"          RMSE: {m['rmse']:.4f}")
    print(f"          R²: {m['r2']:.4f}")


if __name__ == "__main__":
    main()
