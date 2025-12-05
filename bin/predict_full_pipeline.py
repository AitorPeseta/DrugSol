#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_full_pipeline.py
------------------------
Realiza inferencia sobre nuevas moléculas usando el artefacto FINAL_PRODUCT.
Pasos:
1. Carga los datos procesados (Mordred y RDKit/SMILES).
2. Ejecuta predicciones base (GBM, GNN, TPSA).
3. Combina usando el Meta-Modelo (Stacking/Blending).
"""

import argparse
import sys
import os
import json
import pickle
import subprocess
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def run_cmd(cmd):
    print(f"[Exec] CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_gbm(model_dir, X_df):
    """Carga XGBoost y LightGBM y promedia sus predicciones (o usa solo uno si falta otro)"""
    preds = []
    
    # XGBoost
    xgb_path = model_dir / "gbm" / "xgb.pkl"
    if xgb_path.exists():
        print(f"[Exec] Prediciendo con XGBoost...")
        model = load_pickle(xgb_path)
        # XGBoost Pipeline espera DataFrame con nombres correctos
        preds.append(model.predict(X_df))
    
    # LightGBM
    lgb_path = model_dir / "gbm" / "lgbm.pkl"
    if lgb_path.exists():
        print(f"[Exec] Prediciendo con LightGBM...")
        model = load_pickle(lgb_path)
        preds.append(model.predict(X_df))
    
    if not preds:
        raise FileNotFoundError("No se encontraron modelos GBM en FINAL_PRODUCT/base_models/gbm")
    
    # Promedio simple de GBMs (Nivel 0 interno)
    return np.mean(preds, axis=0)

def predict_tpsa(model_dir, df_rdkit):
    """Carga el modelo Baseline (Ridge) y predice"""
    tpsa_path = model_dir / "tpsa" / "tpsa_ridge.pkl" # Ajusta nombre si es distinto
    if not tpsa_path.exists():
        # Intenta buscar cualquier .pkl en esa carpeta
        pkls = list((model_dir / "tpsa").glob("*.pkl"))
        if pkls: tpsa_path = pkls[0]
        else: raise FileNotFoundError("No se encontró modelo TPSA")

    print(f"[Exec] Prediciendo con TPSA Baseline...")
    model = load_pickle(tpsa_path)
    
    # Seleccionar features requeridas por TPSA (definidas en train_full_tpsa)
    # Normalmente: mw, logp, tpsa, temp
    # El modelo es un Pipeline, así que se encarga de seleccionar si se le pasa el DF correcto
    # Pero necesitamos mapear nombres si difieren. 
    # Asumimos que df_rdkit trae: "rdkit__TPSA", "rdkit__logP", "rdkit__MW", "temp_C"
    
    # Renombrar para asegurar compatibilidad si el modelo espera nombres cortos
    # (Depende de cómo entrenaste TPSA, pero el pipeline suele ser robusto)
    return model.predict(df_rdkit)

def predict_gnn(model_dir, input_csv, output_csv, batch_size=50):
    """Invoca a Chemprop para predecir"""
    print(f"[Exec] Prediciendo con Chemprop (GNN)...")
    checkpoint_dir = model_dir / "gnn"
    
    # Buscamos el .pt
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No hay checkpoints .pt en {checkpoint_dir}")
    
    # Usamos el modelo principal (el último o model.pt)
    model_path = [c for c in checkpoints if "model.pt" in c.name]
    model_path = model_path[0] if model_path else checkpoints[0]

    cmd = [
        "chemprop_predict",
        "--test_path", str(input_csv),
        "--preds_path", str(output_csv),
        "--checkpoint_path", str(model_path),
        "--batch_size", str(batch_size),
        "--smiles_columns", "smiles_neutral" # Asegurado por Curate
    ]
    # Si tienes GPU en inferencia:
    # cmd += ["--gpu", "0"]
    
    run_cmd(cmd)
    
    # Leer resultados
    df_res = pd.read_csv(output_csv)
    # Asumimos que la columna target es la única predicción o se llama 'logS'
    # Chemprop devuelve columnas con el nombre del target.
    # Buscamos columna numérica
    pred_cols = [c for c in df_res.columns if c != "smiles_neutral"]
    return df_res[pred_cols[0]].values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-gbm", required=True, help="Parquet con features Mordred")
    ap.add_argument("--data-gnn", required=True, help="Parquet/CSV con SMILES y features RDKit")
    ap.add_argument("--final-product", required=True, help="Directorio FINAL_PRODUCT")
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    prod_dir = Path(args.final_product)
    base_models_dir = prod_dir / "base_models"
    
    # 1. Cargar Datos
    print("[Exec] Cargando datos de entrada...")
    df_gbm = pd.read_parquet(args.data_gbm)
    df_gnn = pd.read_parquet(args.data_gnn) # Contiene SMILES y RDKit features
    
    # Asegurar orden (usando ID si existe, o asumiendo mismo orden si vienen del pipeline)
    # Asumimos mismo orden fila a fila.

    # 2. Predicciones Base
    # A. GBM
    pred_gbm = predict_gbm(base_models_dir, df_gbm)
    
    # B. TPSA (Usa df_gnn que tiene rdkit features)
    pred_tpsa = predict_tpsa(base_models_dir, df_gnn)
    
    # C. GNN (Necesita CSV temporal con SMILES)
    tmp_smiles = "temp_gnn_input.csv"
    tmp_preds  = "temp_gnn_output.csv"
    df_gnn[["smiles_neutral"]].to_csv(tmp_smiles, index=False)
    pred_gnn = predict_gnn(base_models_dir, tmp_smiles, tmp_preds)
    
    # Limpieza temporales
    for f in [tmp_smiles, tmp_preds]:
        if os.path.exists(f): os.remove(f)

    # 3. Ensamblaje (Meta-Modelo)
    print("[Exec] Realizando ensamblaje final...")
    
    # Cargar Model Card para saber estrategia
    with open(prod_dir / "model_card.json") as f:
        card = json.load(f)
    strategy = card.get("strategy", "stack")
    
    # Construir Matriz X para el meta-modelo
    # ORDEN IMPORTANTE: Debe coincidir con como se entrenó (build_final_ensemble)
    # En build_final: sorted(pred_cols) -> ['oof_chemprop', 'oof_lgbm', 'oof_tpsa', 'oof_xgb']
    # Pero aquí tenemos 'pred_gbm' que ya es mezcla de xgb/lgbm si train_methods lo hizo así
    # OJO: Si train_methods usó 4 columnas independientes (xgb, lgbm, gnn, tpsa), 
    # aquí debemos replicarlo.
    
    # REVISIÓN CRÍTICA: En train_methods tu stack recibe [xgb, lgbm, gnn, tpsa].
    # Mi función predict_gbm arriba promediaba. ERROR. 
    # Debemos devolver xgb y lgbm por separado para el stacker.
    
    # --- CORRECCIÓN GBM ---
    xgb_path = base_models_dir / "gbm" / "xgb.pkl"
    lgb_path = base_models_dir / "gbm" / "lgbm.pkl"
    
    y_xgb = load_pickle(xgb_path).predict(df_gbm)
    y_lgb = load_pickle(lgb_path).predict(df_gbm)
    # ----------------------

    # Stack DataFrame
    # Nombres deben coincidir con lo que espera el Ridge o los pesos
    # Si usaste RidgeCV, él no mira nombres de columnas si le pasas numpy, 
    # pero el orden es VITAL.
    # Orden alfabético de columnas OOF era: oof_chemprop, oof_lgbm, oof_tpsa, oof_xgb
    
    X_meta = pd.DataFrame({
        "y_chemprop": pred_gnn,
        "y_lgbm": y_lgb,
        "y_tpsa": pred_tpsa,
        "y_xgb": y_xgb
    })
    
    # Reordenar alfabéticamente para coincidir con el training (y_chemprop, y_lgbm, ...)
    X_meta = X_meta.reindex(sorted(X_meta.columns), axis=1)
    
    final_pred = None
    
    if strategy == "stack":
        with open(prod_dir / "meta_ridge.pkl", "rb") as f:
            stack_data = pickle.load(f)
            # stack_data es un dict {'coef': ..., 'intercept': ...} o el modelo raw
            
        if isinstance(stack_data, dict):
            # Reconstrucción manual
            coef = np.array(stack_data["coef"])
            intercept = stack_data["intercept"]
            final_pred = np.dot(X_meta.values, coef.T) + intercept
        else:
            # Es el objeto sklearn
            final_pred = stack_data.predict(X_meta.values)
            
    elif strategy == "blend":
        with open(prod_dir / "weights.json") as f:
            weights = json.load(f)
        # Weights: {"y_xgb": 0.2, ...}
        final_pred = np.zeros(len(df_gbm))
        for col, w in weights.items():
            if col in X_meta.columns:
                final_pred += X_meta[col].values * w
    
    # 4. Guardar Resultado
    df_out = pd.DataFrame()
    if "row_uid" in df_gbm.columns: df_out["id"] = df_gbm["row_uid"]
    df_out["smiles"] = df_gnn["smiles_neutral"]
    df_out["predicted_logS"] = final_pred
    
    # Añadir componentes por si interesa verlos
    df_out["pred_xgb"] = y_xgb
    df_out["pred_lgbm"] = y_lgb
    df_out["pred_gnn"] = pred_gnn
    df_out["pred_tpsa"] = pred_tpsa

    df_out.to_csv(args.output, index=False)
    print(f"[Exec] Predicciones guardadas en {args.output}")

if __name__ == "__main__":
    main()