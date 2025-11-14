#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

PHENOL_CANDIDATES = ["n_phenol","has_phenol","phenol_like","n_phenol_like"]

def _read_any(p: str) -> pd.DataFrame:
    path = Path(p)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")

def _pick_phenol_col(df: pd.DataFrame):
    for c in PHENOL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _metrics(y, yhat):
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return {"rmse": float(rmse), "r2": float(r2), "n": int(len(y))}

def main():
    ap = argparse.ArgumentParser("OOF TPSA+phenol (Ridge)")
    ap.add_argument("--train", required=True, help="CSV/Parquet con train")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="oof_tpsa")
    ap.add_argument("--folds-file", default=None, help="(opcional) CSV/Parquet con columnas [id-col, fold]")
    ap.add_argument("--kfold", type=int, default=5, help="n_splits si no pasas folds_file")
    ap.add_argument("--seed", type=int, default=42)
    return_ = ap.parse_args()

    outdir = Path(return_.save_dir); outdir.mkdir(parents=True, exist_ok=True)

    df = _read_any(return_.train).copy()
    for c in [return_.id_col, return_.target, "TPSA"]:
        if c not in df.columns:
            raise ValueError(f"Falta columna en train: {c}")

    phenol_col = _pick_phenol_col(df)
    have_tempC = "temp_C" in df.columns

    # construimos X de forma flexible
    X_cols = ["TPSA"]
    if phenol_col: X_cols.append(phenol_col)
    if have_tempC: X_cols.append("temp_C")
    # opcional: raíz de TPSA (suaviza no-linealidad simple)
    df["_sqrt_TPSA"] = np.sqrt(np.clip(df["TPSA"].astype(float), a_min=0, a_max=None))
    X_cols.append("_sqrt_TPSA")

    # limpiar filas con NA en los usados
    use_cols = [return_.id_col, return_.target] + X_cols
    df = df[use_cols].dropna().reset_index(drop=True)

    # folds
    if return_.folds_file:
        folds = _read_any(return_.folds_file)[[return_.id_col, "fold"]]
        df = df.merge(folds, on=return_.id_col, how="inner")
        if df["fold"].isna().any():
            raise ValueError("Hay filas sin fold tras merge con folds_file.")
        unique_folds = sorted(df["fold"].unique().tolist())
        fold_assign = [(df["fold"] == k) for k in unique_folds]
    else:
        kf = KFold(n_splits=return_.kfold, shuffle=True, random_state=return_.seed)
        fold_assign = []
        for _, va_idx in kf.split(df):
            m = np.zeros(len(df), dtype=bool)
            m[va_idx] = True
            fold_assign.append(m)
        unique_folds = list(range(len(fold_assign)))
        df["fold"] = -1  # se asignará abajo

    # modelo base
    alphas = (1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100)
    pipe_cv = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridgecv", RidgeCV(alphas=alphas, store_cv_values=False))
    ])

    y = df[return_.target].to_numpy(float)
    X = df[X_cols].to_numpy(float)

    oof = np.full(len(df), np.nan, dtype=float)
    alpha_by_fold = {}

    for k, va_mask in zip(unique_folds, fold_assign):
        tr_mask = ~va_mask
        if not return_.folds_file:
            df.loc[va_mask, "fold"] = k

        pipe_cv.fit(X[tr_mask], y[tr_mask])
        alpha_k = float(pipe_cv.named_steps["ridgecv"].alpha_)
        alpha_by_fold[int(k)] = alpha_k

        # re-fit con alpha_k fijo
        mdl = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha_k, random_state=return_.seed))
        ])
        mdl.fit(X[tr_mask], y[tr_mask])
        oof[va_mask] = mdl.predict(X[va_mask])

    # métricas OOF
    m_all = _metrics(y, oof)
    (outdir / "metrics_oof_tpsa.json").write_text(json.dumps(m_all, indent=2))

    # guardar en formato común
    common = pd.DataFrame({
        "id": df[return_.id_col].astype("string"),
        "fold": df["fold"].astype(int),
        "y_true": y,
        "y_pred": oof,
        "model": "tpsa"
    })
    common.to_parquet(outdir / "oof_tpsa.parquet", index=False)

    # manifiesto
    manifest = {
        "id_col": return_.id_col,
        "target": return_.target,
        "X_cols": X_cols,
        "alphas_grid": list(alphas),
        "alpha_by_fold": alpha_by_fold,
        "phenol_col_used": phenol_col,
        "has_tempC": have_tempC
    }
    (outdir / "tpsa_oof_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("[OK] OOF TPSA guardado en:", (outdir / "oof_tpsa.parquet").resolve())
    print("[OK] Métricas OOF:", m_all)

if __name__ == "__main__":
    main()
