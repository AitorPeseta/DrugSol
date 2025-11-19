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

PHENOL_CANDIDATES = ["aroOHdel", "n_phenol", "n_phenol_like",
                     "has_phenol", "phenol_like"]


def _read_any(p: str) -> pd.DataFrame:
    path = Path(p)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")


def _ensure_phenol_descriptor(df: pd.DataFrame, forced: str | None = None):
    """
    Intenta construir el descriptor tipo aroOHdel:
      aroOHdel = n_phenol + n_phenol_like

    Prioridades:
      1) forced en df -> usar esa columna tal cual.
      2) 'aroOHdel' ya existe -> usarla.
      3) n_phenol y/o n_phenol_like -> crear aroOHdel = suma.
      4) cualquier candidato en PHENOL_CANDIDATES.
    """
    if forced is not None:
        if forced not in df.columns:
            raise ValueError(f"--phenol-col='{forced}' no existe en el train.")
        return df, forced

    if "aroOHdel" in df.columns:
        return df, "aroOHdel"

    have_nphen = "n_phenol" in df.columns
    have_nphen_like = "n_phenol_like" in df.columns

    if have_nphen or have_nphen_like:
        nphen = (pd.to_numeric(df["n_phenol"], errors="coerce")
                 if have_nphen else 0)
        nphen_like = (pd.to_numeric(df["n_phenol_like"], errors="coerce")
                      if have_nphen_like else 0)
        df["aroOHdel"] = (
            (nphen if hasattr(nphen, "__array__") else 0)
            + (nphen_like if hasattr(nphen_like, "__array__") else 0)
        )
        return df, "aroOHdel"

    for c in PHENOL_CANDIDATES:
        if c in df.columns:
            return df, c

    return df, None


def main():
    ap = argparse.ArgumentParser(
        "FULL TPSA+phenol+mp (Ridge) -> pkl + json en espacio crudo"
    )
    ap.add_argument("--train", required=True)
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="models_TPSA")
    ap.add_argument("--seed", type=int, default=42)

    # Columna de TPSA (para compatibilidad con tu NF)
    ap.add_argument("--tpsa-col", default="TPSA")
    # Columna de fenoles opcional (si no, se construye aroOHdel)
    ap.add_argument("--phenol-col", default=None,
                    help="Si se indica, fuerza esa columna como descriptor de fenoles")

    # argumentos extra que ahora se ignoran pero los aceptamos
    ap.add_argument("--use-tempC", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--smiles-col", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = _read_any(args.train).copy()

    # Validaciones básicas
    needed = [args.target, args.tpsa_col, "logP", "mp"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Falta columna en train: {c}")

    # Fenoles
    df, phenol_col = _ensure_phenol_descriptor(df, forced=args.phenol_col)

    have_mp = "mp" in df.columns
    if have_mp:
        df["mp_m25"] = pd.to_numeric(df["mp"], errors="coerce") - 25.0

    # Features del modelo tipo Phenol: logP, TPSA, mp-25, aroOHdel
    X_cols = ["logP", args.tpsa_col]
    if have_mp:
        X_cols.append("mp_m25")
    if phenol_col:
        X_cols.append(phenol_col)

    use_cols = [args.target] + X_cols
    base = df[use_cols].dropna().reset_index(drop=True)

    y = base[args.target].to_numpy(float)
    X = base[X_cols].to_numpy(float)

    alphas = (1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100)
    ridgecv = RidgeCV(alphas=alphas, store_cv_values=False)
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", ridgecv)])
    pipe.fit(X, y)

    # Guardamos el pipeline completo por si quieres depurar
    joblib.dump(pipe, outdir / "tpsa.pkl")

    manifest = {"feature_cols": X_cols}
    (outdir / "gbm_manifest.json").write_text(json.dumps(manifest, indent=2))

    # -------- Exportar modelo en espacio crudo --------
    scaler = pipe.named_steps["scaler"]
    ridge = pipe.named_steps["ridge"]

    w = np.asarray(ridge.coef_, dtype=float)
    b = float(ridge.intercept_)
    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)

    scales_safe = np.where(scales == 0.0, 1.0, scales)

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
            "logp_col": "logP",
            "tpsa_col": args.tpsa_col,
            "phenol_col": phenol_col,
            "used_mp": bool(have_mp),
            "n_train": int(len(base))
        }
    }
    (outdir / "tpsa_model.json").write_text(json.dumps(model_json, indent=2))

    print("[OK] Guardado:", (outdir / "tpsa.pkl").resolve())
    print("[OK] Manifiesto:", (outdir / "gbm_manifest.json").resolve())
    print("[OK] Modelo JSON crudo:", (outdir / "tpsa_model.json").resolve())


if __name__ == "__main__":
    main()
