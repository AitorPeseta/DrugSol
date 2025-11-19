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

PHENOL_CANDIDATES = ["aroOHdel", "n_phenol", "n_phenol_like",
                     "has_phenol", "phenol_like"]


def _read_any(p: str) -> pd.DataFrame:
    path = Path(p)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")


def _ensure_phenol_descriptor(df: pd.DataFrame):
    """
    Construye un descriptor tipo aroOHdel:
      aroOHdel = n_phenol + n_phenol_like

    Devuelve (df_modificado, nombre_col_phenol).
    """
    if "aroOHdel" in df.columns:
        return df, "aroOHdel"

    have_nphen = "n_phenol" in df.columns
    have_nphen_like = "n_phenol_like" in df.columns

    if have_nphen or have_nphen_like:
        nphen = pd.to_numeric(df["n_phenol"], errors="coerce") if have_nphen else 0
        nphen_like = pd.to_numeric(df["n_phenol_like"], errors="coerce") if have_nphen_like else 0
        df["aroOHdel"] = (
            (nphen if hasattr(nphen, "__array__") else 0)
            + (nphen_like if hasattr(nphen_like, "__array__") else 0)
        )
        return df, "aroOHdel"

    for c in PHENOL_CANDIDATES:
        if c in df.columns:
            return df, c

    return df, None


def _metrics(y, yhat):
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return {"rmse": float(rmse), "r2": float(r2), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser("OOF TPSA+phenol+mp (Ridge)")
    ap.add_argument("--train", required=True, help="CSV/Parquet con train")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="oof_tpsa")
    ap.add_argument("--folds-file", default=None,
                    help="(opcional) CSV/Parquet con columnas [id-col, fold]")
    ap.add_argument("--kfold", type=int, default=5,
                    help="n_splits si no pasas folds_file")
    ap.add_argument("--seed", type=int, default=42)

    # argumentos extra por compatibilidad: se ignoran
    ap.add_argument("--smiles-col", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--tpsa-col", default="TPSA", help=argparse.SUPPRESS)
    args = ap.parse_args()

    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _read_any(args.train).copy()

    needed = [args.id_col, args.target, "logP", "TPSA", "mp"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Falta columna en train: {c}")

    # Fenoles → aroOHdel
    df, phenol_col = _ensure_phenol_descriptor(df)

    have_mp = "mp" in df.columns

    if have_mp:
        df["mp_m25"] = pd.to_numeric(df["mp"], errors="coerce") - 25.0

    X_cols = ["logP", "TPSA"]
    if have_mp:
        X_cols.append("mp_m25")
    if phenol_col:
        X_cols.append(phenol_col)

    # limpiamos filas con NA
    use_cols = [args.id_col, args.target] + X_cols
    df = df[use_cols].dropna().reset_index(drop=True)

    # folds
    if args.folds_file:
        folds = _read_any(args.folds_file)[[args.id_col, "fold"]]
        df = df.merge(folds, on=args.id_col, how="inner")
        if df["fold"].isna().any():
            raise ValueError("Hay filas sin fold tras merge con folds_file.")
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
        df["fold"] = -1  # se rellenará abajo

    alphas = (1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100)
    pipe_cv = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridgecv", RidgeCV(alphas=alphas, store_cv_values=False))
    ])

    y = df[args.target].to_numpy(float)
    X = df[X_cols].to_numpy(float)

    oof = np.full(len(df), np.nan, dtype=float)
    alpha_by_fold = {}

    for k, va_mask in zip(unique_folds, fold_assign):
        tr_mask = ~va_mask
        if not args.folds_file:
            df.loc[va_mask, "fold"] = k

        pipe_cv.fit(X[tr_mask], y[tr_mask])
        alpha_k = float(pipe_cv.named_steps["ridgecv"].alpha_)
        alpha_by_fold[int(k)] = alpha_k

        oof[va_mask] = pipe_cv.predict(X[va_mask])

    # métricas OOF
    m_all = _metrics(y, oof)
    (outdir / "metrics_oof_tpsa.json").write_text(json.dumps(m_all, indent=2))

    # formato común OOF
    common = pd.DataFrame({
        "id": df[args.id_col].astype("string"),
        "fold": df["fold"].astype(int),
        "y_true": y,
        "y_pred": oof,
        "model": "tpsa"
    })
    common.to_parquet(outdir / "oof_tpsa.parquet", index=False)

    # manifiesto con info del modelo
    manifest = {
        "id_col": args.id_col,
        "target": args.target,
        "X_cols": X_cols,
        "alphas_grid": list(alphas),
        "alpha_by_fold": alpha_by_fold,
        "phenol_col_used": phenol_col,
        "has_mp": have_mp,
        "n_rows_valid": int(len(df))
    }
    (outdir / "tpsa_oof_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("[OK] OOF TPSA guardado en:", (outdir / "oof_tpsa.parquet").resolve())
    print("[OK] Métricas OOF:", m_all)


if __name__ == "__main__":
    main()
