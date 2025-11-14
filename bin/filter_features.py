#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Set, Dict

import numpy as np
import pandas as pd

# Modelos opcionales
_HAS_LGBM = False
_HAS_XGB  = False
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    pass

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    pass


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(p)
    raise SystemExit(f"Formato no soportado: {p.suffix}")


def write_like_input(df: pd.DataFrame, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower()
    if suf == ".parquet":
        df.to_parquet(p, index=False)
    elif suf in (".csv", ".txt"):
        df.to_csv(p, index=False)
    else:
        # por defecto parquet
        out_parquet = str(p) + ".parquet"
        df.to_parquet(out_parquet, index=False)


def drop_constant_and_nzv(X: pd.DataFrame, nzv_thresh: float = 0.01) -> pd.DataFrame:
    """Elimina columnas constantes y near-zero variance."""
    Xn = X.select_dtypes(include=[np.number])
    keep = []
    for c in Xn.columns:
        col = Xn[c]
        # constante
        if col.nunique(dropna=True) <= 1:
            continue
        # near-zero variance (proporción de la clase mayoritaria muy alta)
        vc = col.value_counts(dropna=True)
        major_ratio = (vc.iloc[0] / float(len(col))) if len(vc) else 1.0
        if major_ratio >= (1.0 - nzv_thresh):
            # si el 99% de filas tienen un único valor ≈ no informativa
            continue
        keep.append(c)
    return Xn[keep]


def correlation_clusters(X: pd.DataFrame, corr_thresh: float) -> List[List[str]]:
    """Forma clústeres de alta correlación usando un grafo (threshold sobre |rho|)."""
    if X.shape[1] <= 1:
        return [X.columns.tolist()]
    corr = X.corr().abs().fillna(0.0)
    cols = corr.columns.tolist()
    # grafo por umbral
    adj: Dict[str, Set[str]] = {c: set() for c in cols}
    for i, ci in enumerate(cols):
        for j in range(i+1, len(cols)):
            cj = cols[j]
            if corr.iloc[i, j] >= corr_thresh:
                adj[ci].add(cj)
                adj[cj].add(ci)
    # BFS para componentes conexas
    visited = set()
    clusters = []
    for c in cols:
        if c in visited:
            continue
        comp = []
        stack = [c]
        visited.add(c)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        clusters.append(sorted(comp))
    return clusters


def quick_gain_importance(
    X: pd.DataFrame,
    y: pd.Series,
    algo: str = "auto",
    time_limit: int = 60,
    seed: int = 42
) -> pd.Series:
    """
    Entrena un modelo muy rápido y devuelve importancia por 'gain'.
    Si no hay target o falla el modelo -> devuelve varianza normalizada como respaldo.
    """
    if y is None or y.isna().all():
        # respaldo: varianza
        var = X.var(ddof=0).fillna(0.0)
        scl = var.sum() or 1.0
        return (var / scl)

    algo = algo.lower()
    use_lgb = (algo in ("auto", "lgbm")) and _HAS_LGBM
    use_xgb = (algo in ("auto", "xgb")) and _HAS_XGB and (not use_lgb)

    try:
        if use_lgb:
            dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
            params = dict(
                objective="regression",
                metric="rmse",
                learning_rate=0.05,
                num_leaves=64,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                min_data_in_leaf=10,
                seed=seed,
                verbose=-1,
                device_type="cpu",
            )
            bst = lgb.train(params, dtrain, num_boost_round=200)
            imp = bst.feature_importance(importance_type="gain")
            s = pd.Series(imp, index=X.columns, dtype=float)
            if s.sum() == 0:
                # fallback a split count
                imp = bst.feature_importance(importance_type="split")
                s = pd.Series(imp, index=X.columns, dtype=float)
            tot = s.sum() or 1.0
            return s / tot

        if use_xgb:
            dtrain = xgb.DMatrix(X, label=y)
            params = dict(
                objective="reg:squarederror",
                tree_method="hist",
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                seed=seed,
                verbosity=0,
            )
            bst = xgb.train(params, dtrain, num_boost_round=200)
            score = bst.get_score(importance_type="gain")
            s = pd.Series(score, dtype=float)
            # ensure all columns present
            s = s.reindex(X.columns).fillna(0.0)
            tot = s.sum() or 1.0
            return s / tot

    except Exception:
        pass

    # respaldo: varianza
    var = X.var(ddof=0).fillna(0.0)
    scl = var.sum() or 1.0
    return (var / scl)


def select_medoids_by_gain(
    X: pd.DataFrame,
    y: pd.Series,
    clusters: List[List[str]],
    algo: str,
) -> List[str]:
    """Para cada clúster, elige 1 medoid según importancia media (gain); 
       si el clúster tiene 1, lo deja tal cual."""
    selected = []
    # Importancia global como guía; si falla, varianza
    global_imp = quick_gain_importance(X, y, algo=algo)

    for group in clusters:
        if len(group) == 1:
            selected.append(group[0])
            continue
        sub = global_imp.reindex(group).fillna(0.0)
        # si todas cero, usar varianza como desempate
        if sub.sum() == 0:
            var = X[group].var(ddof=0).fillna(0.0)
            sub = var / (var.sum() or 1.0)
        # medoid = argmax de la importancia/varianza
        medoid = sub.sort_values(ascending=False).index[0]
        selected.append(medoid)
    return selected


def front_columns(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    front = [c for c in order if c in df.columns]
    rest  = [c for c in df.columns if c not in front]
    return df[front + rest]


def main():
    ap = argparse.ArgumentParser("Filtra Mordred manteniendo TODO lo demás")
    ap.add_argument("-i", "--input", required=True, help="Entrada .parquet o .csv")
    ap.add_argument("-o", "--output", required=True, help="Salida .parquet o .csv")
    ap.add_argument("--target", default=None, help="Columna target (para importancia de ganancia)")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--mordred-prefix", default="mordred__")
    ap.add_argument("--corr-thresh", type=float, default=0.99)
    ap.add_argument("--nzv-thresh",  type=float, default=0.01)
    ap.add_argument("--algo", choices=["auto", "lgbm", "xgb"], default="auto")
    ap.add_argument("--time-limit", type=int, default=60)
    ap.add_argument("--force-keep", nargs="*", default=[], help="Columnas a blindar (no filtrar/quitar)")

    args = ap.parse_args()

    df = read_any(args.input)
    all_cols = df.columns.tolist()

    # Separar Mordred vs Resto
    mordred_cols = [c for c in all_cols if c.startswith(args.mordred_prefix)]
    passthrough_cols = [c for c in all_cols if c not in mordred_cols]

    # Blindaje de columnas marcadas
    force_keep = set(args.force_keep or [])
    passthrough_cols = list(dict.fromkeys(list(force_keep) + passthrough_cols))

    # Trabajar SOLO en Mordred numéricos
    X_mord = df[mordred_cols].select_dtypes(include=[np.number])
    if X_mord.shape[1] == 0:
        # No hay Mordred numéricos -> devolvemos igual
        out = df.copy()
        out = front_columns(out, [args.id-col if hasattr(args, "id-col") else args.id_col, args.target or "", "smiles_neutral", "smiles_original"])
        write_like_input(out, args.output)
        return

    # 1) constantes / NZV
    X_step1 = drop_constant_and_nzv(X_mord, nzv_thresh=args.nzv_thresh)
    if X_step1.shape[1] == 0:
        # Nada útil -> devuelve solo passthrough
        out = df[passthrough_cols].copy()
        out = front_columns(out, [args.id_col, args.target or "", "smiles_neutral", "smiles_original"])
        write_like_input(out, args.output)
        return

    # 2) clústeres por alta correlación
    clusters = correlation_clusters(X_step1, corr_thresh=args.corr_thresh)

    # 3) seleccionar medoid por importancia de ganancia
    y = None
    if args.target and args.target in df.columns:
        y = pd.to_numeric(df[args.target], errors="coerce")

    keep_mord = select_medoids_by_gain(X_step1, y, clusters, algo=args.algo)

    # Recombinar: TODO lo no-Mordred + Mordred filtradas
    out = pd.concat([df[passthrough_cols], X_step1[keep_mord]], axis=1)

    # Orden “front” útil
    out = front_columns(out, [args.id_col, args.target or "", "smiles_neutral", "smiles_original"])

    write_like_input(out, args.output)
    print(f"[OK] Mordred in: {len(mordred_cols)}  -> kept: {len(keep_mord)}")
    print(f"[OK] Passthrough cols: {len(passthrough_cols)}")
    print(f"[OK] Guardado en: {args.output}")


if __name__ == "__main__":
    main()
