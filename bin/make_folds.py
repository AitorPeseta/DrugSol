#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_folds.py (Robust Version)
------------------------------
Genera 5 folds garantizando que NINGUNA fila se quede sin fold.
Estrategia: StratifiedKFold con fallback a KFold simple para casos difíciles.
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from collections import Counter

def read_any(path):
    if path.endswith(".parquet"): return pd.read_parquet(path)
    return pd.read_csv(path)

def get_strat_label(df, target_col, n_bins=5):
    """Crea bines del target para estratificar."""
    if target_col not in df.columns:
        return np.zeros(len(df), dtype=int)
    
    y = pd.to_numeric(df[target_col], errors='coerce')
    # Intentamos qcut (cuantiles)
    try:
        return pd.qcut(y, n_bins, labels=False, duplicates='drop').fillna(-1).astype(int)
    except:
        # Fallback a cut (ancho fijo) si hay muchos repetidos
        return pd.cut(y, n_bins, labels=False, duplicates='drop').fillna(-1).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--group-col", default="cluster_ecfp4_0p7")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # Argumentos legacy ignorados pero aceptados para compatibilidad
    ap.add_argument("--strat-mode", default="both")
    ap.add_argument("--temp-col", default="temp_C")
    ap.add_argument("--temp-step", default=2)
    ap.add_argument("--temp-unit", default="auto")
    ap.add_argument("--bins", default=5)

    args = ap.parse_args()
    
    df = read_any(args.input)
    
    # 1. Preparar datos de agrupación
    if args.group_col not in df.columns:
        # Si no hay grupos (no debería pasar), cada fila es un grupo
        df["_grp"] = df.index
    else:
        df["_grp"] = df[args.group_col].fillna("UNKNOWN")

    # 2. Preparar etiqueta de estratificación (Solo por Target logS para simplificar)
    # Estratificar por Temp+LogS en 5 folds es demasiado restrictivo y causa los drops.
    # Solo LogS es suficiente para CV.
    df["_strata"] = get_strat_label(df, args.target, n_bins=5)

    # 3. Colapsar a nivel de Grupo (Scaffold)
    # Queremos asignar el fold al GRUPO, no a la molécula
    grp_meta = df.groupby("_grp")["_strata"].agg(lambda x: Counter(x).most_common(1)[0][0]).reset_index()
    
    groups = grp_meta["_grp"].values
    labels = grp_meta["_strata"].values
    
    # 4. Limpieza de clases raras
    # Si una clase tiene menos miembros que n_splits, StratifiedKFold fallará.
    # Las convertimos a una clase "OTHER" (-1)
    counts = Counter(labels)
    labels_clean = np.array([l if counts[l] >= args.n_splits else -1 for l in labels])

    # 5. Split (Con Fallback)
    folds_map = {}
    
    try:
        # Intento 1: Estratificado
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_gen = skf.split(groups, labels_clean)
        mode = "Stratified"
    except Exception:
        # Fallback: Aleatorio simple (pero respetando grupos)
        skf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_gen = skf.split(groups)
        mode = "Random (Fallback)"

    # Asignar folds
    for fold_idx, (_, test_idx) in enumerate(split_gen):
        for idx in test_idx:
            grp_id = groups[idx]
            folds_map[grp_id] = fold_idx

    # 6. Mapear de vuelta a las filas
    df["fold"] = df["_grp"].map(folds_map)
    
    # 7. RED DE SEGURIDAD: Rellenar huerfanos
    # Si por alguna razón matemática extraña alguno quedó NaN
    if df["fold"].isna().any():
        n_missing = df["fold"].isna().sum()
        print(f"[WARN] {n_missing} filas se quedaron sin fold. Asignando aleatoriamente.")
        # Asignar aleatoriamente entre 0..4
        rng = np.random.default_rng(args.seed)
        df.loc[df["fold"].isna(), "fold"] = rng.integers(0, args.n_splits, size=n_missing)

    df["fold"] = df["fold"].astype(int)

    # Chequeo final
    dist = df["fold"].value_counts(normalize=True).sort_index()
    print(f"[Folds] Generados {args.n_splits} folds ({mode}). Distribución:")
    print(dist)
    
    # Guardar
    # Guardamos row_uid, fold, y group_id por si acaso
    out_cols = [args.id_col, "fold"]
    if args.group_col in df.columns: out_cols.append(args.group_col)
    
    df[out_cols].to_parquet(args.out, index=False)

if __name__ == "__main__":
    main()