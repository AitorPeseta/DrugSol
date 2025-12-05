#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_chemprop.py (Pure Graph Version)
-------------------------------------------
Entrena con el 100% de los datos usando SOLO SMILES.
Mantiene la consistencia con OOF.
"""

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem

def run_cmd(cmd):
    print(f"[CMD] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True)

def _read(path):
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)

def is_valid_smiles(s):
    try: return Chem.MolFromSmiles(str(s)) is not None
    except: return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="models/chemprop")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--best-params", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--val-fraction", default=0.05)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    
    outdir = Path(args.save_dir)
    if outdir.exists(): shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Full] Cargando {args.train}...")
    df = _read(args.train)

    valid = df[args.smiles_col].apply(is_valid_smiles)
    df = df[valid].reset_index(drop=True)
    
    # Física: Calculamos pesos pero NO features
    if "temp_C" in df.columns:
        t = df["temp_C"].fillna(25.0)
        df["sw_temp37"] = 1.0 + 2.0 * np.exp(-((t - 37.0) ** 2) / (2 * 8.0**2))
    else:
        df["sw_temp37"] = 1.0

    # Guardar CSVs limpios (Solo SMILES + Target + Peso)
    cols_to_save = [args.smiles_col, args.target]
    
    df[cols_to_save].to_csv(outdir/"data.csv", index=False)
    df[["sw_temp37"]].to_csv(outdir/"weights.csv", index=False, header=True)
    df.iloc[:5][cols_to_save].to_csv(outdir/"val.csv", index=False)
    
    hp = json.loads(Path(args.best_params).read_text())
    
    cmd = [
        "chemprop_train",
        "--data_path", str(outdir/"data.csv"),
        "--separate_val_path", str(outdir/"val.csv"),
        "--separate_test_path", str(outdir/"val.csv"),
        "--data_weights_path", str(outdir/"weights.csv"),
        "--dataset_type", "regression",
        "--save_dir", str(outdir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--smiles_columns", args.smiles_col,
        "--target_columns", args.target,
        "--metric", "rmse",
        
        "--hidden_size", str(int(hp.get("message_hidden_dim", 300))),
        "--depth", str(int(hp.get("depth", 3))),
        "--dropout", str(float(hp.get("dropout", 0.0))),
        "--ffn_num_layers", str(int(hp.get("ffn_num_layers", 2)))
    ]
    
    if args.checkpoint and Path(args.checkpoint).exists():
        cmd += ["--checkpoint_path", str(args.checkpoint)]

    if args.gpu: cmd += ["--gpu", "0"]
        
    run_cmd(cmd)
    
    # Manifest (Sin descriptores)
    (outdir / "chemprop_manifest.json").write_text(json.dumps({
        "target": args.target,
        "descriptors": [],
        "pretrained": bool(args.checkpoint)
    }, indent=2))
    
    print(f"[Full] Terminado.")

if __name__ == "__main__":
    main()