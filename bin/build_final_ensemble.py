#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_final_ensemble.py
-----------------------
Entrena el modelo final (Stacking/Blending) usando TODOS los datos OOF acumulados.
Requiere el archivo de entrenamiento original para obtener los valores reales (logS).
"""

import argparse
import os
import shutil
import json
import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import RidgeCV
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--oof-files", nargs='+', required=True) 
    ap.add_argument("--train-file", required=True, help="Archivo Parquet original con logS y id/row_uid")
    ap.add_argument("--gbm-dir", required=True)
    ap.add_argument("--gnn-dir", required=True)
    ap.add_argument("--tpsa-model", required=True)
    ap.add_argument("--save-dir", default="final_product")

    args = ap.parse_args()
    outdir = Path(args.save_dir)
    
    print(f"[Build] Construyendo Producto Final. Estrategia: {args.strategy.upper()}")
    
    # 1. Copiar Modelos Base
    print("[Build] Copiando modelos base...")
    (outdir / "base_models").mkdir(parents=True, exist_ok=True)
    
    if os.path.isdir(args.gbm_dir):
        shutil.copytree(args.gbm_dir, outdir/"base_models/gbm", dirs_exist_ok=True)
    
    if os.path.isdir(args.gnn_dir):
        shutil.copytree(args.gnn_dir, outdir/"base_models/gnn", dirs_exist_ok=True)
    
    if os.path.isdir(args.tpsa_model):
        shutil.copytree(args.tpsa_model, outdir/"base_models/tpsa", dirs_exist_ok=True)

    # 2. Generar Meta-Modelo
    if args.strategy in ["blend", "stack"]:
        print(f"[Build] Cargando {len(args.oof_files)} archivos OOF...")
        
        # A. Cargar Predicciones OOF
        dfs = []
        for f in args.oof_files:
            dfs.append(pd.read_parquet(f))
        df_oof = pd.concat(dfs, ignore_index=True)
        
        # B. Cargar Ground Truth (logS)
        print(f"[Build] Cargando Ground Truth desde {args.train_file}...")
        df_train = pd.read_parquet(args.train_file)
        
        # --- CORRECCIÓN: Normalizar nombre de ID ---
        # Si existe row_uid pero no id, lo renombramos
        if 'id' not in df_train.columns and 'row_uid' in df_train.columns:
            print("[Build] Renombrando columna 'row_uid' a 'id' para consistencia...")
            df_train = df_train.rename(columns={'row_uid': 'id'})
        # -------------------------------------------

        # Validar columnas
        if 'id' not in df_train.columns or 'logS' not in df_train.columns:
            raise ValueError(f"El archivo de train debe tener columnas 'id' (o 'row_uid') y 'logS'. Encontrado: {df_train.columns}")
            
        # C. MERGE: Unir predicciones con valores reales
        print("[Build] Cruzando OOF con valores reales (Merge por ID)...")
        # Nos quedamos solo con id y logS del train para no duplicar columnas
        df = pd.merge(df_oof, df_train[['id', 'logS']], on='id', how='inner')
        
        print(f"[Build] Dataset Meta-Entrenamiento Final: {len(df)} filas.")
        
        # Detectar columnas de predicción
        pred_cols = [c for c in df.columns if c.startswith("oof_") or c.startswith("y_pred_")]
        # Filtrar columnas que no sean de modelos base
        pred_cols = [c for c in pred_cols if "blend" not in c and "stack" not in c]
        pred_cols = sorted(pred_cols)
        print(f"[Build] Features (Modelos Base): {pred_cols}")
        
        # Definir X e y
        target_col = "logS"
        X = df[pred_cols].values
        y = df[target_col].values
        
        # Limpiar NaNs
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        if args.strategy == "blend":
            print("[Build] Optimizando pesos (NNLS)...")
            from scipy.optimize import nnls
            weights, _ = nnls(X, y)
            if weights.sum() > 0: weights /= weights.sum()
            
            w_dict = {col.replace("oof_", "y_").replace("y_pred_", "y_"): float(w) for col, w in zip(pred_cols, weights)}
            
            with open(outdir / "weights.json", "w") as f:
                json.dump(w_dict, f, indent=2)
            print(f"[Build] Pesos guardados: {w_dict}")
            
        elif args.strategy == "stack":
            print("[Build] Entrenando Stacker (RidgeCV)...")
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])
            model.fit(X, y)
            
            stack_obj = {
                "feature_names": [c.replace("oof_", "y_").replace("y_pred_", "y_") for c in pred_cols],
                "coef": list(model.coef_),
                "intercept": float(model.intercept_)
            }
            with open(outdir / "meta_ridge.pkl", "wb") as f:
                pickle.dump(stack_obj, f)
            print("[Build] Stacker guardado.")

    # Metadatos
    info = {
        "strategy": args.strategy,
        "description": "Modelo final DrugSol v1.0",
        "input_columns": ["smiles", "temperature_K", "solvent"]
    }
    with open(outdir / "model_card.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[Build] ¡Producto final listo en {outdir}!")

if __name__ == "__main__":
    main()