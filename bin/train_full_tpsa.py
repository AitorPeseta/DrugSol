#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
import joblib

PHENOL_CANDIDATES = ["has_phenol","n_phenol","phenol_like","n_phenol_like"]

def _read_any(p: str) -> pd.DataFrame:
    path = Path(p)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")

def _pick_phenol(df: pd.DataFrame):
    for c in PHENOL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser("FULL TPSA+phenol (Ridge) -> pkl + json en espacio crudo")
    ap.add_argument("--train", required=True)
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="models_TPSA")
    ap.add_argument("--seed", type=int, default=42)
    # opcional: forzar columna de fenol y permitir renombrar TPSA
    ap.add_argument("--tpsa-col", default="TPSA")
    ap.add_argument("--phenol-col", default=None)
    ap.add_argument("--use-tempC", action="store_true",
                    help="Si está presente temp_C y activas este flag, la usará como feature adicional")
    args = ap.parse_args()

    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = _read_any(args.train).copy()

    # Validaciones básicas
    if args.target not in df.columns:
        raise ValueError(f"Falta columna target: {args.target}")
    if args.tpsa_col not in df.columns:
        raise ValueError(f"Falta columna TPSA: {args.tpsa_col}")

    # Selección de columnas
    phenol_col = args.phenol_col or _pick_phenol(df)
    have_tempC = ("temp_C" in df.columns) and args.use_tempC

    X_cols = [args.tpsa_col]
    if phenol_col: X_cols.append(phenol_col)
    if have_tempC: X_cols.append("temp_C")

    # sqrt(TPSA) como en el paper
    sqrt_col = f"_sqrt_{args.tpsa_col}"
    df[sqrt_col] = np.sqrt(np.clip(df[args.tpsa_col].astype(float), a_min=0, a_max=None))
    X_cols.append(sqrt_col)

    # Construcción del dataset limpio
    use_cols = [args.target] + X_cols
    base = df[use_cols].dropna().reset_index(drop=True)

    y = base[args.target].to_numpy(float)
    X = base[X_cols].to_numpy(float)

    # Ridge con selección de alpha por CV (sobre datos estandarizados)
    alphas = (1e-3,1e-2,1e-1,1,3,10,30,100)
    ridgecv = RidgeCV(alphas=alphas, store_cv_values=False)
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", ridgecv)])
    pipe.fit(X, y)

    # Guardar pipeline completo (útil para reproducibilidad/diagnóstico)
    joblib.dump(pipe, outdir / "tpsa.pkl")

    # Manifiesto de columnas (para tu helper de GBM si lo quisieras reutilizar)
    manifest = {"feature_cols": X_cols}
    (outdir / "gbm_manifest.json").write_text(json.dumps(manifest, indent=2))

    # -------- Exportar modelo en espacio crudo (JSON) --------
    scaler = pipe.named_steps["scaler"]
    ridge  = pipe.named_steps["ridge"]

    w = np.asarray(ridge.coef_, dtype=float)           # coeficientes sobre X estandarizado
    b = float(ridge.intercept_)                        # intercepto del Ridge
    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)

    # Evitar divisiones por cero si algún scale es 0 (columna constante)
    scales_safe = np.where(scales == 0.0, 1.0, scales)

    # Transformación a espacio crudo:
    # y = b + sum_i w_i * (x_i - mean_i)/scale_i
    #   = (b - sum_i w_i * mean_i/scale_i) + sum_i (w_i/scale_i) * x_i
    raw_coefs = (w / scales_safe)
    raw_intercept = float(b - np.sum(w * means / scales_safe))

    coef_map = {col: float(c) for col, c in zip(X_cols, raw_coefs)}

    model_json = {
        "intercept": raw_intercept,
        "coef": coef_map,
        "features": X_cols,
        "standardized_training": True,
        "alpha_selected": float(getattr(ridge, "alpha_", np.nan)),
        "meta": {
            "target": args.target,
            "tpsa_col": args.tpsa_col,
            "phenol_col": phenol_col,
            "used_tempC": bool(have_tempC),
            "n_train": int(len(base))
        }
    }
    (outdir / "tpsa_model.json").write_text(json.dumps(model_json, indent=2))

    print("[OK] Guardado:", (outdir / "tpsa.pkl").resolve())
    print("[OK] Manifiesto:", (outdir / "gbm_manifest.json").resolve())
    print("[OK] Modelo JSON crudo:", (outdir / "tpsa_model.json").resolve())

if __name__ == "__main__":
    main()
