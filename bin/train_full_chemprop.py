#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_chemprop.py (v1.6.1 Robust Strategy)
-----------------------------------------------
Retrains Chemprop on the full dataset (100% train).
Uses 'Dummy Validation' trick to allow weights without index errors.
"""

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem # Para validar SMILES

def run_cmd(cmd):
    print(f"[CMD] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True)

def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Formato no soportado: {p.suffix}")

def is_valid_smiles(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return m is not None
    except:
        return False

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
    # Args de compatibilidad (ignorados pero necesarios para que no falle la llamada)
    ap.add_argument("--val-fraction", type=float, default=0.05) 
    
    args = ap.parse_args()
    
    outdir = Path(args.save_dir)
    if outdir.exists(): shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar Datos
    print(f"[Full] Loading {args.train}...")
    df = _read(args.train)

    # SANEAMIENTO: Eliminar SMILES inválidos antes de nada
    valid_mask = df[args.smiles_col].apply(is_valid_smiles)
    if (~valid_mask).sum() > 0:
        print(f"[WARN] Eliminando {(~valid_mask).sum()} SMILES inválidos.")
        df = df[valid_mask].reset_index(drop=True)
    
    # 2. Ingeniería Física (Pesos y Temperatura)
    # ------------------------------------------
    if "temp_C" in df.columns:
        t = df["temp_C"].fillna(25.0)
        # Pesos Gaussianos (centrados en 37C)
        df["sw_temp37"] = 1.0 + 2.0 * np.exp(-((t - 37.0) ** 2) / (2 * 8.0**2))
        
        # Feature Termodinámica (1/T)
        if "inv_temp" not in df.columns:
             df["inv_temp"] = 1000.0 / (t + 273.15)
    else:
        print("[WARN] No temp_C column. Weights set to 1.0")
        df["sw_temp37"] = 1.0
    
    # 3. Preparar Archivos (Estrategia Dummy Val)
    # -------------------------------------------
    # Chemprop v1 con pesos falla si intentamos hacer split automático aleatorio.
    # Solución: Darle explícitamente Train (Todo) y Val (Dummy).
    
    # Columnas a guardar
    cols_data = [args.smiles_col, args.target]
    # Descriptores: todo lo numérico que no sea meta
    exclude = [args.smiles_col, args.target, "sw_temp37", "row_uid", "fold", "smiles", "mol"]
    desc_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Nombres de archivo
    data_csv = outdir / "full_train.csv"
    weights_csv = outdir / "full_weights.csv"
    val_dummy_csv = outdir / "val_dummy.csv"
    
    # A) Guardar Train COMPLETO
    df[cols_data + desc_cols].to_csv(data_csv, index=False)
    # Guardar pesos (HEADER=True es más seguro en v1 moderna)
    df[["sw_temp37"]].to_csv(weights_csv, index=False, header=True)
    
    # B) Guardar Val DUMMY (copia de las primeras 5 filas)
    # Esto satisface a Chemprop pero no afecta al modelo final (porque usamos el modelo entrenado en Train)
    df.iloc[:5][cols_data + desc_cols].to_csv(val_dummy_csv, index=False)
    
    # 4. Cargar Hiperparámetros
    hp = json.loads(Path(args.best_params).read_text())
    
    # 5. Comando de Entrenamiento
    cmd = [
        "chemprop_train",
        "--data_path", str(data_csv),
        "--separate_val_path", str(val_dummy_csv), # Bypass split bug
        "--separate_test_path", str(val_dummy_csv),
        "--data_weights_path", str(weights_csv),
        
        "--dataset_type", "regression",
        "--save_dir", str(outdir),
        "--epochs", str(args.epochs),
        "--smiles_columns", args.smiles_col,
        "--target_columns", args.target,
        "--metric", "rmse",
        
        # Hiperparámetros Optimizados
        "--hidden_size", str(int(hp.get("message_hidden_dim", 300))), # Usar .get para seguridad
        "--depth", str(int(hp.get("depth", 3))),
        "--dropout", str(float(hp.get("dropout", 0.0))),
        "--ffn_num_layers", str(int(hp.get("ffn_num_layers", 2)))
    ]
    
    if desc_cols:
        cmd += ["--no_features_scaling"]

    if args.gpu:
        cmd += ["--gpu", "0"]
        
    run_cmd(cmd)
    
    # 6. Manifest
    manifest = {
        "target": args.target,
        "smiles_col": args.smiles_col,
        "v1_legacy": True,
        "descriptors": desc_cols,
        "n_train": len(df)
    }
    (outdir / "chemprop_manifest.json").write_text(json.dumps(manifest, indent=2))
    
    # Limpieza de archivos temporales
    if data_csv.exists(): data_csv.unlink()
    if weights_csv.exists(): weights_csv.unlink()
    if val_dummy_csv.exists(): val_dummy_csv.unlink()

    print(f"[Full] Done. Model saved to {outdir}")

if __name__ == "__main__":
    main()