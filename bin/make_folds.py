#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _qbin(y, q=5):
    """Devuelve bins por cuantiles (0..q-1). Hace fallback si hay empates/NA."""
    y = pd.Series(y).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan)
    if y.notna().sum() < q:
        # Demasiados NAs o pocos valores: bins uniformes
        return pd.cut(y, q, labels=False, duplicates="drop")
    try:
        return pd.qcut(y, q, labels=False, duplicates="drop")
    except Exception:
        return pd.cut(y, q, labels=False, duplicates="drop")


def main():
    ap = argparse.ArgumentParser(
        description="Genera folds OOF por grupos (cluster_ecfp4_0p7) con estratificación por bins del target."
    )
    ap.add_argument("--input", "-i", required=True,
                    help="Train .parquet o .csv con columnas: row_uid, cluster_ecfp4_0p7, logS (por defecto).")
    ap.add_argument("--out", "-o", default="folds.parquet",
                    help="Salida con columnas: row_uid, fold, group_id, y_bin. (default: folds.parquet)")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--group-col", default="cluster_ecfp4_0p7")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=5, help="#bins del target (qcut) si estrat-mode incluye 'target'.")
    ap.add_argument("--strat-mode", choices=["target", "temp", "both"], default="temp",
                    help="Criterio de estratificación a nivel de grupo.")
    ap.add_argument("--temp-col", default=None, help="Columna de temperatura (°C o K).")
    ap.add_argument("--temp-step", type=float, default=5.0, help="Ancho del bin de temperatura (en °C).")
    ap.add_argument("--temp-unit", choices=["auto", "c", "k"], default="auto",
                    help="Unidad de temp-col: auto intenta detectar K si mediana>150.")
    args = ap.parse_args()

    # --- carga ---
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        raise ValueError("Entrada debe ser .parquet o .csv")

    required = [args.id_col, args.group_col]
    if args.strat_mode in ("target", "both"):
        required.append(args.target)
    if args.strat_mode in ("temp", "both"):
        if args.temp_col is None:
            raise ValueError("--temp-col es obligatorio cuando strat-mode incluye 'temp'")
        required.append(args.temp_col)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col}")

    # --- etiquetas de estratificación a nivel de grupo ---
    grp = df[[args.group_col]].drop_duplicates().copy()

    # 1) Target -> y_bin por cuantiles
    if args.strat_mode in ("target", "both"):
        y_med = (df.groupby(args.group_col, dropna=False)[args.target]
                   .median()
                   .rename("y_med"))
        grp = grp.merge(y_med, on=args.group_col, how="left")
        grp["y_bin"] = _qbin(grp["y_med"], q=args.bins)
        if grp["y_bin"].isna().all():
            grp["y_bin"] = 0

    # 2) Temperatura -> T_bin por step (en °C)
    if args.strat_mode in ("temp", "both"):
        t = pd.to_numeric(df[args.temp_col], errors="coerce")
        unit = args.temp_unit
        if unit == "auto":
            unit = "k" if (pd.Series(t).median() or 0) > 150 else "c"
        if unit == "k":
            t_c = t - 273.15
        else:
            t_c = t
        df["_Tbin"] = (t_c / args.temp_step).round() * args.temp_step
        T_mode = (df.groupby(args.group_col, dropna=False)["_Tbin"]
                    .agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
                    .rename("T_bin"))
        grp = grp.merge(T_mode, on=args.group_col, how="left")
        grp["T_bin"] = grp["T_bin"].fillna("OTHER")

    # 3) Composición final de la etiqueta de estrato por grupo
    if args.strat_mode == "target":
        grp["glabel"] = grp["y_bin"].astype(str)
    elif args.strat_mode == "temp":
        grp["glabel"] = grp["T_bin"].astype(str)
    else:  # both
        grp["glabel"] = grp["y_bin"].astype(str) + "|" + grp["T_bin"].astype(str)

    # Filtramos grupos válidos
    grp_valid = grp[grp[args.group_col].notna()].copy()
    if grp_valid.empty:
        raise ValueError("No hay grupos válidos para hacer folds.")

    groups = grp_valid[args.group_col].to_numpy()
    labels = grp_valid["glabel"].astype(str).to_numpy()

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    # Asignamos un fold por grupo
    fold_by_group = {}
    for fold, (_, val_idx) in enumerate(skf.split(groups, labels)):
        for gi in groups[val_idx]:
            fold_by_group[gi] = fold

    # Mapear a filas
    out = df[[args.id_col, args.group_col]].copy()
    out = out.rename(columns={args.group_col: "group_id"})
    out["fold"] = out["group_id"].map(fold_by_group)

    # Por seguridad: quita filas cuyo grupo no cayó en ningún fold (p.ej. NaN)
    out = out.dropna(subset=["fold"]).copy()
    out["fold"] = out["fold"].astype(int)

    # Añadimos el bin de su grupo (útil para auditoría)
    # Anexamos etiqueta para auditoría
    out = out.merge(grp_valid[[args.group_col, "glabel"]]
                    .rename(columns={args.group_col: "group_id"}),
                    on="group_id", how="left")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[OK] Guardado: {out_path.resolve()}")
    print(out["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
