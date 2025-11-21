#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenamiento FULL del modelo 'Physically Inspired Ridge'.
Sustituye la lógica de fenoles por MW (proxy de energía cristalina) 
e inv_temp (termodinámica).

Exporta:
1. .pkl (Pipeline de scikit-learn)
2. .json (Pesos crudos des-estandarizados para inferencia portátil)
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
import joblib

def _read_any(p: str) -> pd.DataFrame:
    path = Path(p)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")

def main():
    ap = argparse.ArgumentParser(
        "FULL Ali-GSE-Mod (Ridge) -> pkl + json en espacio crudo"
    )
    ap.add_argument("--train", required=True)
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="models_TPSA")
    ap.add_argument("--seed", type=int, default=42)

    # Columnas físicas necesarias
    ap.add_argument("--tpsa-col", default="TPSA", help="Columna de superficie polar")
    ap.add_argument("--mw-col", default="MW", help="Columna de Peso Molecular (Proxy de MP)")
    ap.add_argument("--temp-col", default="temp_C", help="Columna de temperatura en Celsius")
    
    # Argumentos legacy (se mantienen para que no rompa si tu pipeline los envía)
    ap.add_argument("--phenol-col", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--smiles-col", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar Datos
    df = _read_any(args.train).copy()

    # 2. Validaciones y Preparación de Features Físicos
    # Necesitamos LogP, TPSA, MW
    needed_cols = ["logP", args.tpsa_col, args.mw_col]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Falta columna crítica en train: {c}")

    # 3. Ingeniería de Temperatura (Termodinámica: 1/T)
    # Si no hay columna de temp, asumimos 25°C para no romper, pero avisamos.
    if args.temp_col in df.columns:
        # Rellenar nulos con 25.0 por seguridad
        temps = df[args.temp_col].fillna(25.0)
        # Ecuación: 1000 / (T_kelvin)
        df["inv_temp"] = 1000.0 / (temps + 273.15)
    else:
        print(f"[WARN] No se encontró '{args.temp_col}'. Asumiendo 25°C constante.")
        df["inv_temp"] = 1000.0 / (25.0 + 273.15)

    # Lista final de features para el modelo
    # Orden: [Hidrofobicidad, Polaridad, Tamaño/Sólido, Temperatura]
    X_cols = ["logP", args.tpsa_col, args.mw_col, "inv_temp"]

    # 4. Limpieza final
    use_cols = [args.target] + X_cols
    base = df[use_cols].dropna().reset_index(drop=True)
    
    if len(base) == 0:
        raise ValueError("El dataset quedó vacío tras dropna(). Revisa tus datos.")

    print(f"Entrenando con {len(base)} muestras y features: {X_cols}")

    y = base[args.target].to_numpy(float)
    X = base[X_cols].to_numpy(float)

    # 5. Entrenamiento (RidgeCV con Estandarización)
    alphas = (1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100)
    ridgecv = RidgeCV(alphas=alphas, store_cv_values=False)
    
    # Pipeline: Importante estandarizar porque MW (~300) y inv_temp (~3.3) tienen escalas muy distintas
    pipe = Pipeline([
        ("scaler", StandardScaler()), 
        ("ridge", ridgecv)
    ])
    pipe.fit(X, y)

    # 6. Guardado del objeto Python (Pickle)
    joblib.dump(pipe, outdir / "tpsa_phys.pkl")

    # Manifiesto simple para saber qué columnas entran
    manifest = {"feature_cols": X_cols}
    (outdir / "tpsa_manifest.json").write_text(json.dumps(manifest, indent=2))

    # 7. Exportación matemática ("Raw Weights")
    # Esto permite usar la fórmula: y = Intercept + C1*LogP + C2*TPSA + ...
    # sin necesitar el objeto StandardScaler de sklearn.
    scaler = pipe.named_steps["scaler"]
    ridge = pipe.named_steps["ridge"]

    w = np.asarray(ridge.coef_, dtype=float)
    b = float(ridge.intercept_)
    
    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    # Evitar división por cero (aunque StandardScaler maneja vars constantes, mejor prevenir)
    scales_safe = np.where(scales == 0.0, 1.0, scales)

    # Des-estandarización de coeficientes:
    # w_raw = w_scaled / sigma
    # b_raw = b_scaled - sum(w_scaled * mu / sigma)
    raw_coefs = (w / scales_safe)
    raw_intercept = float(b - np.sum(w * means / scales_safe))

    coef_map = {col: float(c) for col, c in zip(X_cols, raw_coefs)}

    model_json = {
        "model_type": "Ridge_GSE_Approx",
        "intercept": raw_intercept,
        "coef": coef_map,
        "features": X_cols,
        "formula_help": "Prediction = intercept + sum(val[f] * coef[f])",
        "standardized_training": True,
        "alpha_selected": float(getattr(ridge, "alpha_", np.nan)),
        "meta": {
            "target": args.target,
            "logp_col": "logP",
            "tpsa_col": args.tpsa_col,
            "mw_col": args.mw_col,
            "temp_mode": "inv_temp_kelvin",
            "n_train": int(len(base))
        }
    }
    
    # Guardamos como tpsa_model.json para mantener compatibilidad de nombres si lo necesitas
    (outdir / "tpsa_model.json").write_text(json.dumps(model_json, indent=2))

    print("[OK] Guardado Pipeline:", (outdir / "tpsa.pkl").resolve())
    print("[OK] Guardado JSON Pesos:", (outdir / "tpsa_model.json").resolve())
    print(f"[INFO] Intercepto base (aprox): {raw_intercept:.4f}")

if __name__ == "__main__":
    main()