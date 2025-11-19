#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, pickle, math
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# =============== Métricas ===============

def root_mse(y, yhat):
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y, yhat))
    except Exception:
        return float(math.sqrt(mean_squared_error(y, yhat)))

def r2(y, yhat):
    return float(r2_score(y, yhat))

# =============== Utilidades generales ===============

def read_any(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path} (usa .parquet o .csv)")

def ensure_same_length(arrs, names):
    lens = [len(a) for a in arrs]
    if len(set(lens)) != 1:
        raise ValueError("Longitudes distintas: " + ", ".join(f"{n}={l}" for n, l in zip(names, lens)))

def collapse_oof(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa duplicados por (id, fold): y_true=first, y_pred=mean.
    Aquí 'id' debe ser tu row_uid (compuesto+solvente+T).
    """
    req = {"id", "fold", "y_true", "y_pred"}
    if not req.issubset(df.columns):
        raise ValueError(f"OOF sin columnas requeridas {sorted(req)} -> columnas={list(df.columns)}")
    g = df.groupby(["id", "fold"], as_index=False)
    return g.agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))

def count_dup_pairs(df: pd.DataFrame) -> int:
    return int((df.groupby(["id","fold"]).size() > 1).sum())

def check_y_true_consistency(merged: pd.DataFrame, tol: float = 1e-8):
    """
    Tras merge de múltiples OOF (cada uno renombrado), comprobamos que y_true sea consistente.
    Permitimos pequeñas diferencias numéricas (tol).
    """
    y_cols = [c for c in merged.columns if c.startswith("y_true_")]
    if not y_cols:
        return
    base = merged[y_cols[0]].to_numpy()
    for c in y_cols[1:]:
        diff = np.nanmax(np.abs(base - merged[c].to_numpy()))
        if not np.isfinite(diff):
            raise ValueError("Valores no finitos en y_true durante validación.")
        if diff > tol:
            raise ValueError(f"Inconsistencia en y_true entre OOFs (max |Δ|={diff:.3e} > {tol})")

def simple_merge_on_id_fold(dfs_named):
    """
    Une varios dataframes colapsados por (id,fold) evitando multiplicidad.
    A cada df se le renombran columnas a: y_true_<name>, oof_<name>.
    Devuelve merged, labels.
    """
    merged = None
    labels = []
    for name, df in dfs_named:
        # colapso defensivo extra (por si llegan sin colapsar)
        if count_dup_pairs(df) > 0:
            df = collapse_oof(df)
        df = df.loc[:, ["id","fold","y_true","y_pred"]].copy()
        df = df.rename(columns={"y_true": f"y_true_{name}", "y_pred": f"oof_{name}"})
        labels.append(name)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["id","fold"], how="inner")
    if merged is None or merged.empty:
        raise ValueError("El merge de OOF por (id, fold) quedó vacío (desalineación).")
    return merged, labels

def extract_y_and_Z_from_merged(merged: pd.DataFrame, labels) -> tuple:
    """
    Selecciona y (a partir de y_true_* validado) y construye Z = columnas oof_<label>.
    """
    y_cols = [f"y_true_{lab}" for lab in labels if f"y_true_{lab}" in merged.columns]
    if not y_cols:
        y_cols = [c for c in merged.columns if c.startswith("y_true_")]
    check_y_true_consistency(merged, tol=1e-8)
    y = merged[y_cols[0]].to_numpy()
    Z = merged[[f"oof_{lab}" for lab in labels]].to_numpy(dtype=float)
    folds = merged["fold"].to_numpy()
    return y, Z, folds

# =============== Blending simplex ===============

def simplex_project(v):
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(n) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

def blend_weights(Z, y, l2=0.0):
    Z = np.asarray(Z, dtype=float); y = np.asarray(y, dtype=float)
    A = Z.T @ Z + l2 * np.eye(Z.shape[1])
    b = Z.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return simplex_project(w)

# =============== Cargas OOF (3 modos) ===============

def load_oof_table_mode(args):
    df = read_any(args.oof_table)
    if "y" not in df.columns:
        raise ValueError(f"{args.oof_table} debe contener columna 'y'")
    y = df["y"].to_numpy()
    cols = args.oof_labels if args.oof_labels else [c for c in df.columns if c != "y"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas OOF en {args.oof_table}: {missing}")
    Z = df[cols].to_numpy(dtype=float)
    return y, Z, cols, None, None, None, {"mode":"A"}

def load_oof_files_mode(args):
    y_df = read_any(args.y_file)
    if "y" in y_df.columns:
        y = y_df["y"].to_numpy()
    else:
        if y_df.shape[1] != 1:
            raise ValueError(f"{args.y_file} debe tener columna 'y' o una sola columna")
        y = y_df.iloc[:, 0].to_numpy()
    if len(args.oof_files) != len(args.oof_labels):
        raise ValueError("--oof-files y --oof-labels deben tener la misma longitud")
    Z_parts = []
    for f, lab in zip(args.oof_files, args.oof_labels):
        df = read_any(f)
        col = df[lab] if lab in df.columns else (df.iloc[:,0] if df.shape[1]==1 else None)
        if col is None:
            raise ValueError(f"{f} debe tener una columna con el nombre {lab} o una sola columna")
        Z_parts.append(col.to_numpy())
    ensure_same_length(Z_parts + [y], args.oof_labels + ["y"])
    Z = np.vstack(Z_parts).T
    return y, Z, list(args.oof_labels), None, None, None, {"mode":"B"}

def load_oof_common_mode(oof_files, labels=None):
    """
    Recomendado. Cada OOF debe tener: id (=row_uid), fold, y_true, y_pred[, model, seed]
    """
    dfs_named = []
    dup_info = {}
    for i, f in enumerate(oof_files):
        df = read_any(f)
        req = {"id", "fold", "y_true", "y_pred"}
        if not req.issubset(df.columns):
            raise ValueError(f"{f} debe contener columnas {sorted(req)}")
        name = (labels[i] if labels and i < len(labels)
                else (df["model"].iloc[0] if "model" in df.columns else f"m{i}"))
        n_dup = count_dup_pairs(df)
        dup_info[name] = int(n_dup)
        if n_dup > 0:
            df = collapse_oof(df)
        dfs_named.append((name, df))

    merged, labels_final = simple_merge_on_id_fold(dfs_named)

    # Validación y extracción
    y, Z, folds = extract_y_and_Z_from_merged(merged, labels_final)

    # Normaliza y_true a una única columna
    y_cols = [c for c in merged.columns if c.startswith("y_true_")]
    merged = merged.drop(columns=[c for c in y_cols if c != y_cols[0]]).rename(columns={y_cols[0]: "y_true"})
    merged = merged[["id","fold","y_true"] + [f"oof_{lab}" for lab in labels_final]]

    meta = {
        "mode":"C",
        "dup_counts_by_model": dup_info,
        "n_rows_raw_sum": int(sum(read_any(f).shape[0] for f in oof_files)),
        "n_pairs_unique_after_merge": int(merged.shape[0]),
    }
    return y, Z, labels_final, "id", folds, merged, meta

# =============== Meta sin fuga (usa folds) ===============

def stacking_oof_with_folds(Z, y, folds, alphas=(1e-3,1e-2,1e-1,1,3,10,30,100), seed=42):
    Z = np.asarray(Z, dtype=float); y = np.asarray(y, dtype=float)
    uniq = np.unique(folds)
    yhat = np.zeros_like(y, dtype=float)
    ridge_cv_full = RidgeCV(alphas=alphas, store_cv_values=False,
                            cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    ridge_cv_full.fit(Z, y)
    alpha_full = float(ridge_cv_full.alpha_)
    for k in uniq:
        va = (folds == k); tr = ~va
        meta_k = RidgeCV(alphas=alphas, store_cv_values=False,
                         cv=KFold(n_splits=5, shuffle=True, random_state=seed))
        meta_k.fit(Z[tr], y[tr])
        yhat[va] = meta_k.predict(Z[va])
    meta_full = Ridge(alpha=alpha_full, random_state=seed).fit(Z, y)
    return yhat, alpha_full, meta_full

def blending_oof_with_folds(Z, y, folds, l2=1e-8):
    Z = np.asarray(Z, dtype=float); y = np.asarray(y, dtype=float)
    uniq = np.unique(folds)
    yhat = np.zeros_like(y, dtype=float)
    for k in uniq:
        va = (folds == k); tr = ~va
        w_k = blend_weights(Z[tr], y[tr], l2=l2)
        yhat[va] = Z[va] @ w_k
    w_full = blend_weights(Z, y, l2=l2)
    return yhat, w_full

# =============== CLI ===============

def parse_args():
    ap = argparse.ArgumentParser(
        description="Paso 2 (OOF): Meta-learning (stacking) y blending sobre salidas OOF"
    )
    # Modos A/B (opcionales)
    ap.add_argument("--oof-table", help="Parquet/CSV con 'y' y columnas de modelos (modo A)")
    ap.add_argument("--y-file", help="Parquet/CSV con la columna 'y' (modo B)")
    ap.add_argument("--oof-files", nargs="*", help="Lista de ficheros OOF (modo B)")
    ap.add_argument("--oof-labels", nargs="*", help="Nombres/etiquetas de modelos (modo A/B)")

    # Modo C (recomendado): OOFs en formato común
    ap.add_argument("--oof-common", nargs="*", help="OOFs con esquema (id, fold, y_true, y_pred[,…])")
    ap.add_argument("--labels", nargs="*", help="Etiquetas para --oof-common (si se omiten, usa columna 'model' si existe)")

    ap.add_argument("--metric", choices=["rmse", "r2"], default="rmse",
                    help="Criterio para elegir entre stacking y blending (default: rmse)")
    ap.add_argument("--save-dir", default="meta_results", help="Carpeta de salida")
    ap.add_argument("--suffix", default="", help="Sufijo para nombres de salida, p.ej. _v1")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# =============== Main ===============

def main():
    args = parse_args()
    outdir = Path(args.save_dir)
    (outdir / "stack").mkdir(parents=True, exist_ok=True)
    (outdir / "blend").mkdir(parents=True, exist_ok=True)

    meta_info = {}
    # 1) Cargar OOF
    if args.oof_common:
        y, Z, labels, id_col, folds, merged_df, meta_info = load_oof_common_mode(args.oof_common, args.labels)
    elif args.oof_table:
        y, Z, labels, id_col, folds, merged_df, meta_info = load_oof_table_mode(args)
    else:
        y, Z, labels, id_col, folds, merged_df, meta_info = load_oof_files_mode(args)

    print(f"[INFO] Modelos base: {labels} (m={len(labels)}), OOF shape={Z.shape}")

    # 2) Métricas OOF por base
    base_metrics = {lab: {"rmse": root_mse(y, Z[:, j]), "r2": r2(y, Z[:, j])}
                    for j, lab in enumerate(labels)}

    # 3) Stacking OOF
    if folds is not None:
        oof_stack, alpha_full, meta_full = stacking_oof_with_folds(Z, y, folds, seed=args.seed)
        stack_summary = {"alpha_full": alpha_full,
                         "oof_rmse": root_mse(y, oof_stack),
                         "oof_r2": r2(y, oof_stack)}
    else:
        ridge_cv = RidgeCV(alphas=(1e-3,1e-2,1e-1,1,3,10,30,100), store_cv_values=False,
                           cv=KFold(n_splits=5, shuffle=True, random_state=args.seed))
        ridge_cv.fit(Z, y)
        alpha = float(ridge_cv.alpha_)
        meta_full = Ridge(alpha=alpha, random_state=args.seed).fit(Z, y)
        oof_stack = meta_full.predict(Z)
        stack_summary = {"alpha_full": alpha,
                         "oof_rmse": root_mse(y, oof_stack),
                         "oof_r2": r2(y, oof_stack)}

    # 4) Blending OOF
    if folds is not None:
        oof_blend, w_full = blending_oof_with_folds(Z, y, folds, l2=1e-8)
    else:
        w_full = blend_weights(Z, y, l2=1e-8)
        oof_blend = Z @ w_full
    blend_summary = {"weights_full": dict(zip(labels, [float(x) for x in w_full])),
                     "oof_rmse": root_mse(y, oof_blend),
                     "oof_r2": r2(y, oof_blend)}

    # 5) Selección
    if args.metric == "rmse":
        better = "stack" if stack_summary["oof_rmse"] <= blend_summary["oof_rmse"] else "blend"
        stack_score = stack_summary["oof_rmse"]; blend_score = blend_summary["oof_rmse"]
    else:
        better = "stack" if stack_summary["oof_r2"] >= blend_summary["oof_r2"] else "blend"
        stack_score = stack_summary["oof_r2"]; blend_score = blend_summary["oof_r2"]
    print(f"[INFO] Mejor método por {args.metric}: {better} "
          f"(stack={stack_score:.5f}, blend={blend_score:.5f})")

    # 6) Guardados OOF combinados
    if merged_df is not None:
        oof_df = merged_df.copy()
        for j, lab in enumerate(labels):
            oof_df[f"oof_{lab}"] = oof_df[f"oof_{lab}"].astype(float)
        oof_df["oof_stack"] = oof_stack
        oof_df["oof_blend"] = oof_blend
    else:
        oof_df = pd.DataFrame({"y": y}, copy=False)
        for j, lab in enumerate(labels):
            oof_df[f"oof_{lab}"] = Z[:, j]
        oof_df["oof_stack"] = oof_stack
        oof_df["oof_blend"] = oof_blend

    oof_path = outdir / f"oof_predictions{args.suffix}.parquet"
    oof_df.to_parquet(oof_path, index=False)

    # 7) Guardar modelo de STACK en formato que entiende final_infer_master
    #    Mapeamos nombres de modelos -> columnas de nivel 0 (y_xgb, y_lgbm, y_chemprop, y_tpsa, …)
    alias = {
        "xgb": "y_xgb",
        "lgbm": "y_lgbm",
        "gnn": "y_chemprop",
        "chemprop": "y_chemprop",
        "tpsa": "y_tpsa",
    }
    feature_names = [alias.get(lab, f"y_{lab}") for lab in labels]
    stack_obj = {
        "labels": labels,
        "feature_names": feature_names,
        "coef": [float(c) for c in np.asarray(meta_full.coef_, dtype=float).ravel()],
        "intercept": float(meta_full.intercept_),
    }

    with open(outdir / f"stack/meta_ridge{args.suffix}.pkl", "wb") as f:
        pickle.dump(stack_obj, f)

    # 8) Guardar pesos de BLEND
    with open(outdir / f"blend/weights{args.suffix}.json", "w") as f:
        json.dump(blend_summary["weights_full"], f, indent=2)

    # Informe enriquecido
    report = {
        "labels": labels,
        "base_oof": base_metrics,
        "stack": stack_summary,
        "blend": blend_summary,
        "selection": {"metric": args.metric, "better": better,
                      "stack_score": stack_score, "blend_score": blend_score},
        "n_models": len(labels),
        "has_folds": bool(folds is not None)
    }
    if meta_info.get("mode") == "C":
        report.update({
            "mode": "C",
            "dup_counts_by_model": meta_info.get("dup_counts_by_model", {}),
            "n_rows_raw_sum": meta_info.get("n_rows_raw_sum"),
            "n_pairs_unique_after_merge": meta_info.get("n_pairs_unique_after_merge"),
        })
    with open(outdir / f"metrics_oof{args.suffix}.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[DONE] Guardado en: {outdir.resolve()}")
    print(f" - OOF combinado:      {oof_path.name}")
    print(f" - STACK model:         stack/meta_ridge{args.suffix}.pkl")
    print(f" - BLEND weights:       blend/weights{args.suffix}.json")
    print(f" - METRICS (OOF):       metrics_oof{args.suffix}.json")

if __name__ == "__main__":
    main()
