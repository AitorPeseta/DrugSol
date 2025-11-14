#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit


# ---------------- utilidades ----------------

def build_strat_label(df, solvent_col="solvent", temp_col="temperature_K", temp_step=5):
    """
    Crea strat_label = '<solvent>|<temperature_bin>' redondeando temperatura a múltiplos de temp_step.
    """
    if solvent_col not in df.columns:
        raise ValueError(f"Falta la columna solvente: {solvent_col}")
    if temp_col not in df.columns:
        raise ValueError(f"Falta la columna temperatura: {temp_col}")

    temp = df[temp_col]
    if not np.issubdtype(temp.dtype, np.number):
        # intentar convertir
        temp = pd.to_numeric(temp, errors="coerce")

    temp_r = np.round(temp.astype(float) / temp_step) * temp_step
    temp_str = temp_r.apply(lambda x: str(int(x)) if np.isfinite(x) and float(x).is_integer() else str(x))
    return df[solvent_col].astype(str) + "|" + temp_str

def pick_group_label(labels):
    """Devuelve una etiqueta representativa por grupo (moda; en empate, orden alfabético)."""
    c = Counter(labels)
    max_freq = max(c.values())
    candidates = sorted([k for k, v in c.items() if v == max_freq])
    return candidates[0]

def collapse_rare_classes(group_df, label_col="group_strat_label", min_groups_per_class=2):
    """
    Colapsa clases con < min_groups_per_class grupos a 'OTHER'.
    Devuelve (serie colapsada, conteos originales).
    """
    counts = group_df[label_col].value_counts()
    rare = counts[counts < min_groups_per_class].index
    collapsed = group_df[label_col].where(~group_df[label_col].isin(rare), other="OTHER")
    return collapsed, counts.to_dict()


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Split estratificado POR GRUPO usando una columna de grupo precomputada (p.ej. cluster_ecfp4_0p7 o InChIKey14)."
    )
    ap.add_argument("--input", "-i", required=True, help="Entrada .parquet o .csv con la columna de grupo ya creada")
    ap.add_argument("--group-col", required=True, help="Columna de grupo YA EXISTENTE (cluster_ecfp4_0p7, InChIKey14, etc.)")
    ap.add_argument("--temp-col", default="temperature_K")
    ap.add_argument("--temp-step", type=float, default=5.0, help="Paso de redondeo (K) para la estratificación")
    ap.add_argument("--test-size", "-t", type=float, default=0.2)
    ap.add_argument("--seed", "-s", type=int, default=42)
    ap.add_argument("--min-groups-per-class", type=int, default=2, help="Mínimo de GRUPOS por clase para estratificar")
    ap.add_argument("--invalid-group", type=int, default=-1,
                    help="Valor de grupo a excluir antes del split (útil si tu clustering marcó inválidos con -1)")
    ap.add_argument("--save-csv", action="store_true", help="Guardar también train.csv / test.csv")
    args = ap.parse_args()

    # --- carga ---
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        print("La entrada debe ser .parquet o .csv", file=sys.stderr)
        sys.exit(1)

    assert args.group_col in df.columns, f"Falta columna de grupo: {args.group_col}"
    assert args.temp_col in df.columns, f"Falta columna temperatura: {args.temp_col}"

    # --- limpieza ligera de grupos ---
    df_grp = df.copy()
    # si hay inválidos marcados (p.ej. -1), se excluyen del split
    valid_mask = True
    if args.invalid_group is not None and args.group_col in df_grp.columns:
        valid_mask = df_grp[args.group_col] != args.invalid_group
        excl = int((~valid_mask).sum())
        if excl > 0:
            print(f"[INFO] Excluidas {excl} filas con {args.group_col} == {args.invalid_group} (no entran al split).")
    df_grp = df_grp.loc[valid_mask].copy()

    if df_grp.empty:
        raise ValueError("No quedan filas válidas para hacer el split.")

    # --- estrato a nivel de fila ---
    df_grp["strat_label"] = build_strat_label(
        df_grp, temp_col=args.temp_col, temp_step=args.temp_step
    )

    # --- una fila por grupo con una etiqueta representativa ---
    grp = (
        df_grp.groupby(args.group_col)["strat_label"]
              .apply(lambda s: pick_group_label(s.tolist()))
              .reset_index()
              .rename(columns={"strat_label": "group_strat_label"})
    )

    # --- colapso de clases raras y decisión estratificada ---
    grp["group_strat_label"], raw_counts = collapse_rare_classes(
        grp, label_col="group_strat_label", min_groups_per_class=args.min_groups_per_class
    )

    groups_unique = grp[args.group_col].to_numpy()
    y_groups = grp["group_strat_label"].to_numpy()
    n_classes = len(np.unique(y_groups))
    print(f"[INFO] Grupos únicos: {len(groups_unique)} | Clases tras colapso: {n_classes}")

    # --- split por grupos (estratificado si hay ≥2 clases) ---
    idx = np.arange(len(groups_unique))
    if n_classes >= 2:
        print("[INFO] Estratificación por grupos ACTIVADA")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_gi, te_gi = next(splitter.split(idx, y=y_groups))
    else:
        print("[INFO] Estratificación desactivada (solo 1 clase). Fallback a GroupShuffleSplit")
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_gi, te_gi = next(splitter.split(idx, groups=groups_unique))

    train_groups = set(groups_unique[tr_gi])
    test_groups  = set(groups_unique[te_gi])
    assert len(train_groups & test_groups) == 0, "Hay grupos simultáneamente en train y test"

    # --- mapear a filas originales (incluye también las filas excluidas si quieres, aquí NO) ---
    is_train = df[args.group_col].isin(train_groups)
    is_test  = df[args.group_col].isin(test_groups)

    df_train = df.loc[is_train].drop(columns=[c for c in ["strat_label"] if c in df.columns]).reset_index(drop=True)
    df_test  = df.loc[is_test ].drop(columns=[c for c in ["strat_label"] if c in df.columns]).reset_index(drop=True)

    # --- guardar ---
    df_train.to_parquet("train.parquet", index=False)
    df_test.to_parquet("test.parquet", index=False)

    if args.save_csv:
        df_train.to_csv("train.csv", index=False)
        df_test.to_csv("test.csv", index=False)


    # --- resumen ---
    def _group_stats(dfx, name):
        ng = dfx[args.group_col].nunique()
        n = len(dfx)
        print(f"[OK] {name}: filas={n}, grupos={ng}")

    _group_stats(df_train, "TRAIN")
    _group_stats(df_test, "TEST")
    # distribución por clase (si hay)
    if n_classes >= 2:
        m = df_grp[[args.group_col, "strat_label"]].drop_duplicates(args.group_col)
        m["group_strat_label"] = grp.set_index(args.group_col).loc[m[args.group_col], "group_strat_label"].values
        tr_cls = m.loc[m[args.group_col].isin(train_groups), "group_strat_label"].value_counts().to_dict()
        te_cls = m.loc[m[args.group_col].isin(test_groups ), "group_strat_label"].value_counts().to_dict()
        print(f"[OK] Distribución de clases por grupos -> TRAIN: {tr_cls} | TEST: {te_cls}")

    print("[OK] Guardado: train.parquet, test.parquet", (" (y CSV)" if args.save_csv else ""))


if __name__ == "__main__":
    main()
