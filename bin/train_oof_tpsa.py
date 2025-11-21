#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def _read_any(p: str) -> pd.DataFrame:
    path = Path(p)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")

def _metrics(y, yhat):
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return {"rmse": float(rmse), "r2": float(r2), "n": int(len(y))}

def main():
    ap = argparse.ArgumentParser("OOF Modified Ali (Ridge)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="oof_ali_mod")
    ap.add_argument("--folds-file", default=None)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    
    # Nombres de columnas esperadas en tu CSV/Parquet
    ap.add_argument("--mw-col", default="MW", help="Peso molecular")
    ap.add_argument("--temp-col", default="temp_C", help="Temperatura en Celsius")
    
    args = ap.parse_args()
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _read_any(args.train).copy()

    # 1. GESTIÓN DE TEMPERATURA (Crítico si tus datos varían de 25 a 49)
    # La física dice que logS es proporcional a 1/T (Kelvin)
    if args.temp_col in df.columns:
        print(f"Detectada columna temperatura: {args.temp_col}. Creando 1000/T_K")
        # Rellenar nulos con 25°C si los hubiera
        df[args.temp_col] = df[args.temp_col].fillna(25.0)
        df["inv_temp"] = 1000.0 / (df[args.temp_col] + 273.15)
    else:
        print("No se detectó temperatura. Asumiendo constante (no ideal).")
        df["inv_temp"] = 0.0 # El intercepto del Ridge absorberá esto

    # 2. DEFINICIÓN DE FEATURES
    # En lugar de solo fenoles, usamos MW y LogP/TPSA. 
    # Si tienes 'n_rotatable_bonds' o 'aromatic_rings', añádelos aquí también.
    base_features = ["logP", "TPSA"]
    
    # Intentamos añadir MW (Proxy del tamaño/MP)
    if args.mw_col in df.columns:
        base_features.append(args.mw_col)
    
    # Intentamos añadir features estructurales si existen (mejor que solo fenoles)
    # El conteo de fenoles lo puedes dejar, pero no como sustituto único del MP
    if "n_phenol" in df.columns:
        base_features.append("n_phenol")
        
    # Añadimos la temperatura inversa
    final_features = base_features + ["inv_temp"]
    
    # Verificar existencia
    missing = [c for c in final_features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
        
    print(f"Entrenando Ridge con features: {final_features}")

    # Limpieza
    use_cols = [args.id_col, args.target] + final_features
    df = df[use_cols].dropna().reset_index(drop=True)

    # --- Preparación de Folds (Idéntico a tu código) ---
    if args.folds_file:
        folds = _read_any(args.folds_file)[[args.id_col, "fold"]]
        df = df.merge(folds, on=args.id_col, how="inner")
        unique_folds = sorted(df["fold"].unique().tolist())
        fold_assign = [(df["fold"] == k) for k in unique_folds]
    else:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        fold_assign = []
        for _, va_idx in kf.split(df):
            m = np.zeros(len(df), dtype=bool)
            m[va_idx] = True
            fold_assign.append(m)
        unique_folds = list(range(len(fold_assign)))
        df["fold"] = -1

    # --- Pipeline Ridge ---
    # Nota: RidgeCV es excelente aquí. Normalizar es obligatorio.
    alphas = (0.1, 1.0, 10.0, 100.0) 
    pipe_cv = Pipeline([
        ("scaler", StandardScaler()), 
        ("ridgecv", RidgeCV(alphas=alphas)),
    ])

    y = df[args.target].to_numpy(float)
    X = df[final_features].to_numpy(float)

    oof = np.full(len(df), np.nan, dtype=float)
    
    for k, va_mask in zip(unique_folds, fold_assign):
        tr_mask = ~va_mask
        if not args.folds_file: df.loc[va_mask, "fold"] = k

        pipe_cv.fit(X[tr_mask], y[tr_mask])
        oof[va_mask] = pipe_cv.predict(X[va_mask])

    # Métricas y Guardado
    m_all = _metrics(y, oof)
    
    # Guardar outputs
    common = pd.DataFrame({
        "id": df[args.id_col].astype("string"),
        "fold": df["fold"].astype(int),
        "y_true": y,
        "y_pred": oof,
        "model": "ridge_phys", # Nombre actualizado
    })
    
    common.to_parquet(outdir / "oof_tpsa.parquet", index=False)
    (outdir / "metrics_oof_tpsa.json").write_text(json.dumps(m_all, indent=2))
    
    print(f"[OK] RMSE: {m_all['rmse']:.4f} | R2: {m_all['r2']:.4f}")

if __name__ == "__main__":
    main()