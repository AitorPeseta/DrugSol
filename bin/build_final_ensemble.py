#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_final_ensemble.py
-----------------------
Construye el 'Final Product' copiando modelos y entrenando el meta-learner.
CORREGIDO: Ahora copia correctamente el modelo TPSA aunque sea un archivo suelto.
"""

import argparse
import os
import shutil
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import RidgeCV
from pathlib import Path

def copy_model_source(src, dest_folder):
    """Copia un modelo (archivo o carpeta) al destino asegurando que exista"""
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    src_path = Path(src)
    
    if not src_path.exists():
        print(f"[WARN] No se encuentra la fuente del modelo: {src}")
        return

    if src_path.is_dir():
        # Si es directorio, copiamos el contenido recursivamente
        print(f"[Build] Copiando directorio {src_path.name} a {dest_folder}...")
        shutil.copytree(src_path, dest_folder, dirs_exist_ok=True)
    else:
        # Si es archivo, lo copiamos directamente
        print(f"[Build] Copiando archivo {src_path.name} a {dest_folder}...")
        shutil.copy(src_path, dest_folder / src_path.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--oof-files", nargs='+', required=True) 
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--gbm-dir", required=True)
    ap.add_argument("--gnn-dir", required=True)
    ap.add_argument("--tpsa-model", required=True) # Puede ser archivo o carpeta
    ap.add_argument("--save-dir", default="final_product")

    args = ap.parse_args()
    outdir = Path(args.save_dir)
    
    print(f"[Build] Construyendo Producto Final. Estrategia: {args.strategy.upper()}")
    
    # 1. Copiar Modelos Base (Usando la nueva función robusta)
    print("[Build] Empaquetando modelos base...")
    
    copy_model_source(args.gbm_dir,    outdir / "base_models/gbm")
    copy_model_source(args.gnn_dir,    outdir / "base_models/gnn")
    copy_model_source(args.tpsa_model, outdir / "base_models/tpsa")  # <--- AQUÍ FALLABA ANTES

    # 2. Generar Meta-Modelo
    if args.strategy in ["blend", "stack"]:
        print(f"[Build] Cargando {len(args.oof_files)} archivos OOF...")
        
        # A. Cargar OOFs
        dfs = []
        for f in args.oof_files:
            try:
                df_chunk = pd.read_parquet(f)
                dfs.append(df_chunk)
            except Exception as e:
                print(f"[WARN] Error leyendo {f}: {e}")
                
        if not dfs:
            raise ValueError("No se pudieron cargar archivos OOF.")
            
        df_oof = pd.concat(dfs, ignore_index=True)
        
        # B. Cargar Ground Truth
        print(f"[Build] Cargando Ground Truth desde {args.train_file}...")
        df_train = pd.read_parquet(args.train_file)
        
        # Normalizar ID
        if 'id' not in df_train.columns and 'row_uid' in df_train.columns:
            df_train = df_train.rename(columns={'row_uid': 'id'})

        # C. Merge
        if 'id' in df_train.columns and 'logS' in df_train.columns:
            df = pd.merge(df_oof, df_train[['id', 'logS']], on='id', how='inner')
        else:
            print("[WARN] No se pudo hacer merge por ID. Asumiendo orden (PELIGROSO).")
            # Fallback simple si fallan los IDs (solo si longitudes coinciden)
            if len(df_oof) == len(df_train):
                df = df_oof.copy()
                df['logS'] = df_train['logS'].values
            else:
                raise ValueError("Mismatch de IDs y longitudes. No se puede entrenar el Stacker.")

        print(f"[Build] Dataset Meta-Entrenamiento: {len(df)} filas.")
        
        # Detectar columnas de predicción (modelos base)
        pred_cols = [c for c in df.columns if c.startswith("oof_") or c.startswith("y_pred_")]
        # Ignorar columnas del propio ensemble si existen
        pred_cols = [c for c in pred_cols if "blend" not in c and "stack" not in c]
        pred_cols = sorted(pred_cols)
        
        print(f"[Build] Features detectadas: {pred_cols}")
        
        X = df[pred_cols].values
        y = df['logS'].values
        
        # Limpiar NaNs
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        if args.strategy == "blend":
            print("[Build] Optimizando pesos (NNLS)...")
            from scipy.optimize import nnls
            weights, _ = nnls(X, y)
            if weights.sum() > 0: weights /= weights.sum()
            
            # Guardamos mapeo limpio: oof_xgb -> y_xgb
            w_dict = {col.replace("oof_", "y_").replace("y_pred_", "y_"): float(w) for col, w in zip(pred_cols, weights)}
            
            with open(outdir / "weights.json", "w") as f:
                json.dump(w_dict, f, indent=2)
            print(f"[Build] Pesos guardados: {w_dict}")
            
        elif args.strategy == "stack":
            print("[Build] Entrenando Stacker (RidgeCV)...")
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])
            model.fit(X, y)
            
            # Guardamos nombres estandarizados
            clean_names = [c.replace("oof_", "y_").replace("y_pred_", "y_") for c in pred_cols]
            
            stack_obj = {
                "feature_names": clean_names,
                "coef": list(model.coef_),
                "intercept": float(model.intercept_)
            }
            with open(outdir / "meta_ridge.pkl", "wb") as f:
                pickle.dump(stack_obj, f)
            print(f"[Build] Stacker guardado. Coeficientes: {dict(zip(clean_names, model.coef_))}")

    # Metadatos
    info = {
        "strategy": args.strategy,
        "description": "Modelo final DrugSol v1.0",
        "version": "1.0.0"
    }
    with open(outdir / "model_card.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[Build] ¡Producto final generado correctamente en {outdir}!")

if __name__ == "__main__":
    main()