#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_full_pipeline.py
------------------------
Realiza inferencia sobre nuevas moléculas usando el artefacto FINAL_PRODUCT.
Versión corregida: 
1. Mapeo flexible de nombres (y_gnn vs y_chemprop).
2. Búsqueda recursiva de modelos para evitar 'Not Found'.
"""

import argparse
import sys
import os
import json
import pickle
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Ignorar warnings de sklearn sobre versiones o nombres si no son críticos
warnings.filterwarnings("ignore", category=UserWarning)

def run_cmd(cmd):
    print(f"[Exec] CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def find_file(directory, pattern):
    """Busca un archivo recursivamente si no está en la raíz"""
    matches = list(Path(directory).rglob(pattern))
    return matches[0] if matches else None

def filter_model_features(model, df, model_name="Model"):
    """Selecciona SOLO las columnas que el modelo espera."""
    if hasattr(model, "feature_names_in_"):
        required_cols = model.feature_names_in_
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            # print(f"[WARN] {model_name}: Faltan {len(missing)} columnas (ej. {missing[:3]}). Rellenando con 0.")
            for c in missing: df[c] = 0.0
        return df[required_cols]
    
    # Fallback si no tiene metadatos
    metadata_cols = ["row_uid", "smiles", "smiles_neutral", "logS", "InChIKey", "groups"]
    cols_to_keep = [c for c in df.columns if c not in metadata_cols]
    return df[cols_to_keep]

def predict_gbm(model_dir, df_raw):
    """Carga XGBoost y LightGBM y devuelve sus predicciones"""
    y_xgb = None
    y_lgb = None
    
    # Búsqueda flexible de los modelos
    xgb_path = find_file(model_dir / "gbm", "xgb.pkl")
    lgb_path = find_file(model_dir / "gbm", "lgbm.pkl")
    
    # XGBoost
    if xgb_path:
        print(f"[Exec] Prediciendo con XGBoost ({xgb_path.name})...")
        model = load_pickle(xgb_path)
        X_clean = filter_model_features(model, df_raw, "XGBoost")
        y_xgb = model.predict(X_clean)
    
    # LightGBM
    if lgb_path:
        print(f"[Exec] Prediciendo con LightGBM ({lgb_path.name})...")
        model = load_pickle(lgb_path)
        X_clean = filter_model_features(model, df_raw, "LightGBM")
        y_lgb = model.predict(X_clean)
    
    # Fallbacks si falta alguno
    if y_xgb is None and y_lgb is None:
        print("[ERROR] No se encontraron modelos GBM. Devolviendo ceros.")
        return np.zeros(len(df_raw)), np.zeros(len(df_raw))
        
    if y_xgb is None: y_xgb = y_lgb
    if y_lgb is None: y_lgb = y_xgb
    
    return y_xgb, y_lgb

def predict_tpsa(model_dir, df_raw):
    """
    Carga el modelo Baseline. 
    Soporta formato JSON (coeficientes explícitos) y Pickle (objeto sklearn).
    """
    tpsa_dir = model_dir / "tpsa"
    
    # 1. Intentar buscar modelo JSON (input_tpsa o *.json)
    # Buscamos archivos que puedan ser el json
    candidates = list(tpsa_dir.glob("*json")) + list(tpsa_dir.glob("input_tpsa"))
    
    # Filtramos directorios, nos quedamos con archivos
    candidates = [f for f in candidates if f.is_file()]
    
    if candidates:
        model_path = candidates[0]
        print(f"[Exec] Prediciendo con TPSA JSON ({model_path.name})...")
        try:
            with open(model_path, 'r') as f:
                model = json.load(f)
            
            # A. Preparar variables físicas (inv_temp)
            df_calc = df_raw.copy()
            if "temp_C" in df_calc.columns:
                # Kelvin inverso * 1000 (escala típica)
                df_calc["inv_temp"] = 1000.0 / (df_calc["temp_C"] + 273.15)
            else:
                # Default a 25 C si no hay temperatura
                df_calc["inv_temp"] = 1000.0 / 298.15

            # B. Calcular Predicción (Intercept + Sum(Coef * Val))
            intercept = model.get("intercept", 0.0)
            preds = np.full(len(df_calc), intercept)
            
            coefs = model.get("coefs", {})
            features_found = []
            
            for feat, w in coefs.items():
                val = None
                # 1. Buscamos la feature tal cual (ej. rdkit__TPSA)
                if feat in df_calc.columns:
                    val = df_calc[feat]
                # 2. Buscamos sin prefijo (ej. TPSA)
                elif feat.replace("rdkit__", "") in df_calc.columns:
                    val = df_calc[feat.replace("rdkit__", "")]
                
                if val is not None:
                    # Rellenar NaNs con 0 para la suma
                    preds += pd.to_numeric(val, errors='coerce').fillna(0.0).values * w
                    features_found.append(feat)
            
            # print(f"   -> Features usadas: {features_found}")
            return preds

        except Exception as e:
            print(f"[WARN] Falló la carga del JSON TPSA: {e}. Intentando fallback PKL...")

    # 2. Fallback: Buscar modelo Pickle (.pkl)
    pkl_files = list(tpsa_dir.rglob("*.pkl"))
    model_path = next((p for p in pkl_files if "meta" not in p.name), None)

    if not model_path:
        print(f"[WARN] No se encontró modelo TPSA (ni JSON ni PKL) en {tpsa_dir}. Devolviendo ceros.")
        return np.zeros(len(df_raw))

    print(f"[Exec] Prediciendo con TPSA Pickle ({model_path.name})...")
    model = load_pickle(model_path)
    
    # Lógica de renombreado para pickles antiguos
    df_mapped = df_raw.copy()
    rename_map = {c: c.replace("rdkit__", "") for c in df_mapped.columns if c.startswith("rdkit__")}
    if rename_map:
        df_mapped = df_mapped.rename(columns=rename_map)
        
    X_clean = filter_model_features(model, df_mapped, "TPSA")
    return model.predict(X_clean)

def predict_gnn(model_dir, input_csv, output_csv, batch_size=50):
    """Invoca a Chemprop para predecir"""
    print(f"[Exec] Prediciendo con Chemprop (GNN)...")
    checkpoint_dir = model_dir / "gnn"
    
    # Buscar modelo .pt recursivamente
    model_path = find_file(checkpoint_dir, "*.pt")
    
    if not model_path:
        print(f"[WARN] No hay modelo GNN (.pt) en {checkpoint_dir}. Devolviendo ceros.")
        return np.zeros(pd.read_csv(input_csv).shape[0])
    
    cmd = [
        "chemprop_predict",
        "--test_path", str(input_csv),
        "--preds_path", str(output_csv),
        "--checkpoint_path", str(model_path),
        "--batch_size", str(batch_size),
        "--smiles_columns", "smiles_neutral"
    ]
    
    # Chemprop a veces es ruidoso, capturamos error si falla
    try:
        run_cmd(cmd)
        df_res = pd.read_csv(output_csv)
        pred_cols = [c for c in df_res.columns if c != "smiles_neutral" and c != "smiles"]
        return df_res[pred_cols[0]].values
    except Exception as e:
        print(f"[WARN] Falló Chemprop: {e}. Devolviendo ceros.")
        return np.zeros(pd.read_csv(input_csv).shape[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-gbm", required=True)
    ap.add_argument("--data-gnn", required=True)
    ap.add_argument("--final-product", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    prod_dir = Path(args.final_product)
    base_models_dir = prod_dir / "base_models"
    
    # 1. Cargar Datos
    print("[Exec] Cargando datos de entrada...")
    df_gbm = pd.read_parquet(args.data_gbm)
    df_gnn = pd.read_parquet(args.data_gnn) 
    
    # 2. Predicciones Base
    y_xgb, y_lgb = predict_gbm(base_models_dir, df_gbm)
    pred_tpsa = predict_tpsa(base_models_dir, df_gnn)
    
    # GNN (con CSV temporal)
    tmp_smiles = "temp_gnn_input.csv"
    tmp_preds  = "temp_gnn_output.csv"
    
    if "smiles_neutral" not in df_gnn.columns:
        df_gnn["smiles_neutral"] = df_gnn["smiles"] if "smiles" in df_gnn.columns else df_gnn.iloc[:,0]

    df_gnn[["smiles_neutral"]].to_csv(tmp_smiles, index=False)
    pred_gnn = predict_gnn(base_models_dir, tmp_smiles, tmp_preds)
    
    for f in [tmp_smiles, tmp_preds]:
        if os.path.exists(f): os.remove(f)

    # 3. Ensamblaje (Meta-Modelo)
    print("[Exec] Realizando ensamblaje final...")
    
    # Construir DataFrame de Meta-Features
    # IMPORTANTE: Creamos alias para evitar el KeyError 'y_gnn' vs 'y_chemprop'
    X_meta = pd.DataFrame({
        "y_lgbm": y_lgb,
        "y_tpsa": pred_tpsa,
        "y_xgb": y_xgb,
        "y_gnn": pred_gnn,         # Nombre probable 1
        "y_chemprop": pred_gnn,    # Nombre probable 2
        "oof_gnn": pred_gnn,       # Nombre legacy
        "oof_chemprop": pred_gnn   # Nombre legacy
    })
    
    final_pred = None
    
    # Cargar Stacker
    meta_model_path = prod_dir / "meta_ridge.pkl"
    if not meta_model_path.exists():
        print("[WARN] No se encontró meta_ridge.pkl. Usando promedio simple (Blending).")
        final_pred = (y_xgb + y_lgb + pred_gnn + pred_tpsa) / 4
    else:
        with open(meta_model_path, "rb") as f:
            stack_model = pickle.load(f)
            
        # Lógica Robusta: Seleccionar SOLO las columnas que el Stacker quiere
        if isinstance(stack_model, dict) and "feature_names" in stack_model:
            required_cols = stack_model["feature_names"]
            print(f"[Exec] El Stacker espera estas columnas: {required_cols}")
            
            # Verificar si las tenemos todas
            missing = [c for c in required_cols if c not in X_meta.columns]
            if missing:
                print(f"[WARN] Faltan columnas críticas: {missing}. Rellenando con 0.")
                for c in missing: X_meta[c] = 0.0
            
            # Filtrar y ordenar
            X_final = X_meta[required_cols].values
            
            coef = np.array(stack_model["coef"])
            intercept = stack_model["intercept"]
            final_pred = np.dot(X_final, coef.T) + intercept
            
        elif hasattr(stack_model, "predict"):
            # Si es un objeto sklearn directo, intentamos pasarle las columnas que coincidan
            # Esto es más arriesgado si no tiene feature_names_in_
            if hasattr(stack_model, "feature_names_in_"):
                 X_final = X_meta[stack_model.feature_names_in_]
                 final_pred = stack_model.predict(X_final)
            else:
                 # Fallback: Usar orden alfabético de las 4 principales
                 cols_fallback = sorted(["y_xgb", "y_lgbm", "y_tpsa", "y_gnn"])
                 final_pred = stack_model.predict(X_meta[cols_fallback])

    # 4. Guardar Resultado
    df_out = pd.DataFrame()
    df_out["smiles"] = df_gnn["smiles_neutral"]
    df_out["predicted_logS"] = final_pred

    df_out.to_csv(args.output, index=False)
    print(f"[Exec] ¡ÉXITO! Predicciones guardadas en {args.output}")

if __name__ == "__main__":
    main()