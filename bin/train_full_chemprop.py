#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Full Chemprop: Final D-MPNN Model Training
=================================================

Trains the final Chemprop D-MPNN model on the complete dataset using
hyperparameters optimized during OOF cross-validation.

Training Strategy:
    - Pure graph approach: only SMILES structure, no external features
    - Consistent with OOF training for model consistency
    - Sample weighting for temperature importance
    - Minimal validation set (5 samples) to satisfy Chemprop requirements

Arguments:
    --train       : Training data file (Parquet/CSV)
    --smiles-col  : SMILES column name (default: smiles_neutral)
    --target      : Target column name (default: logS)
    --best-params : JSON file with optimized hyperparameters
    --save-dir    : Output directory
    --epochs      : Training epochs (default: 40)
    --batch-size  : Batch size (default: 50)
    --gpu         : Enable GPU acceleration
    --weight-col  : Sample weight column
    --seed        : Random seed

Usage:
    python train_full_chemprop.py \\
        --train train_data.parquet \\
        --smiles-col smiles_neutral \\
        --target logS \\
        --best-params chemprop_best_params.json \\
        --epochs 50 \\
        --gpu \\
        --weight-col weight \\
        --save-dir models_GNN

Output:
    - Chemprop model checkpoint (.pt files)
    - chemprop_manifest.json: Model metadata for inference

Notes:
    - Requires chemprop package
    - GPU recommended for reasonable training times
    - Invalid SMILES are filtered before training
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


# ============================================================================
# UTILITIES
# ============================================================================

def run_command(cmd):
    """Run shell command with logging."""
    print(f"[CMD] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True)


def read_data(path: str) -> pd.DataFrame:
    """Read Parquet or CSV file."""
    p = Path(path)
    if p.suffix == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)


def is_valid_smiles(s):
    """Check if SMILES string is valid."""
    try:
        return Chem.MolFromSmiles(str(s)) is not None
    except Exception:
        return False


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for full Chemprop training."""
    
    ap = argparse.ArgumentParser(
        description="Train full Chemprop D-MPNN model."
    )
    ap.add_argument("--train", required=True,
                    help="Training data file")
    ap.add_argument("--smiles-col", default="smiles_neutral",
                    help="SMILES column name")
    ap.add_argument("--target", default="logS",
                    help="Target column name")
    ap.add_argument("--best-params", required=True,
                    help="JSON file with best hyperparameters")
    ap.add_argument("--save-dir", default="models/chemprop",
                    help="Output directory")
    ap.add_argument("--epochs", type=int, default=40,
                    help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="Batch size")
    ap.add_argument("--gpu", action="store_true",
                    help="Enable GPU")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    ap.add_argument("--weight-col", default=None,
                    help="Sample weight column")
    ap.add_argument("--checkpoint", default=None,
                    help="Resume from checkpoint")
    ap.add_argument("--val-fraction", type=float, default=0.05,
                    help="Validation fraction (not used, for compatibility)")
    
    args = ap.parse_args()
    
    # Setup output directory
    outdir = Path(args.save_dir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load and Validate Data
    # -------------------------------------------------------------------------
    print(f"[Full Chemprop] Loading {args.train}...")
    df = read_data(args.train)
    
    # Filter invalid SMILES
    valid_mask = df[args.smiles_col].apply(is_valid_smiles)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"[Full Chemprop] Filtered {n_invalid} invalid SMILES")
    
    df = df[valid_mask].reset_index(drop=True)
    print(f"[Full Chemprop] Training on {len(df):,} molecules")
    
    # -------------------------------------------------------------------------
    # Handle Sample Weights
    # -------------------------------------------------------------------------
    weight_col_internal = "sw_temp37"  # Internal name for Chemprop CSV
    
    if args.weight_col and args.weight_col in df.columns:
        print(f"[Full Chemprop] Using weights from '{args.weight_col}'")
        df[weight_col_internal] = df[args.weight_col].fillna(1.0)
    else:
        print("[Full Chemprop] No weight column, using uniform weights")
        df[weight_col_internal] = 1.0
    
    # -------------------------------------------------------------------------
    # Prepare CSVs for Chemprop
    # -------------------------------------------------------------------------
    # Pure graph strategy: only SMILES and target
    cols_to_save = [args.smiles_col, args.target]
    
    # Main data
    df[cols_to_save].to_csv(outdir / "data.csv", index=False)
    
    # Weights file
    df[[weight_col_internal]].to_csv(outdir / "weights.csv", index=False, header=True)
    
    # Dummy validation set (Chemprop requires separate val/test)
    df.iloc[:5][cols_to_save].to_csv(outdir / "val.csv", index=False)
    
    # -------------------------------------------------------------------------
    # Load Hyperparameters
    # -------------------------------------------------------------------------
    hp = json.loads(Path(args.best_params).read_text())
    print(f"[Full Chemprop] Hyperparameters: {hp}")
    
    # -------------------------------------------------------------------------
    # Build Chemprop Command
    # -------------------------------------------------------------------------
    cmd = [
        "chemprop_train",
        "--data_path", str(outdir / "data.csv"),
        "--separate_val_path", str(outdir / "val.csv"),
        "--separate_test_path", str(outdir / "val.csv"),
        "--data_weights_path", str(outdir / "weights.csv"),
        "--dataset_type", "regression",
        "--save_dir", str(outdir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--smiles_columns", args.smiles_col,
        "--target_columns", args.target,
        "--metric", "rmse",
        # Hyperparameters
        "--hidden_size", str(int(hp.get("message_hidden_dim", 300))),
        "--depth", str(int(hp.get("depth", 3))),
        "--dropout", str(float(hp.get("dropout", 0.0))),
        "--ffn_num_layers", str(int(hp.get("ffn_num_layers", 2)))
    ]
    
    # Resume from checkpoint
    if args.checkpoint and Path(args.checkpoint).exists():
        cmd += ["--checkpoint_path", str(args.checkpoint)]
    
    # GPU flag
    if args.gpu:
        cmd += ["--gpu", "0"]
    
    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print("[Full Chemprop] Starting training...")
    run_command(cmd)
    
    # -------------------------------------------------------------------------
    # Save Manifest
    # -------------------------------------------------------------------------
    manifest = {
        "model_type": "chemprop_dmpnn",
        "target": args.target,
        "smiles_col": args.smiles_col,
        "hyperparameters": hp,
        "epochs": args.epochs,
        "n_train": len(df),
        "descriptors": [],  # Pure graph, no external descriptors
        "pretrained": bool(args.checkpoint)
    }
    
    with open(outdir / "chemprop_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[Full Chemprop] Done. Model saved to {outdir}/")


if __name__ == "__main__":
    main()
