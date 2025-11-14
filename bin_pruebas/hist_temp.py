#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    elif suf in {".csv", ".tsv"}:
        sep = "," if suf == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Formato no soportado: {suf} (usa .csv o .parquet)")


def guess_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cl = cand.lower()
        if cl in cols_lower:
            return cols_lower[cl]
    # intento por startswith/contains
    for c in df.columns:
        cl = c.lower()
        if any(cl.startswith(x) for x in candidates) or any(x in cl for x in candidates):
            return c
    return None


_num_pat = re.compile(r"[-+]?\d+(\.\d+)?")

def normalize_temperature(s, round_to: float) -> Optional[float]:
    """
    Devuelve temperatura como float (p. ej. 25.0) a partir de:
    - números (25, 30.0)
    - strings con símbolos ("25 C", "25°C", "30 deg")
    Si no encuentra número -> None.
    """
    if pd.isna(s):
        return None
    # si ya es número
    if isinstance(s, (int, float, np.integer, np.floating)):
        val = float(s)
    else:
        m = _num_pat.search(str(s))
        if not m:
            return None
        val = float(m.group(0))
    if round_to is None or round_to <= 0:
        return val
    return round(val / round_to) * round_to


def main():
    ap = argparse.ArgumentParser(description="Histogramas de número de compuestos por temperatura (global y por split).")
    ap.add_argument("--infile", required=True, help="Ruta a CSV/Parquet con los datos.")
    ap.add_argument("--outdir", default=".", help="Carpeta de salida para PNG/CSV.")
    ap.add_argument("--temp-col", default=None,
                    help="Nombre de la columna de temperatura (opcional; si no, se intenta adivinar).")
    ap.add_argument("--split-col", default="split",
                    help="Columna con el split (train/valid/test). Por defecto 'split'.")
    ap.add_argument("--id-col", default=None,
                    help="Columna de ID/SMILES (solo para info; no es necesaria).")
    ap.add_argument("--round", type=float, default=1.0,
                    help="Redondeo de temperatura (p. ej. 1.0 para agrupar 24.7→25). Usa 5.0 si tienes 20/25/30.")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Oculta temperaturas con menos de 'min-count' instancias en gráficos.")
    ap.add_argument("--dpi", type=int, default=180, help="Resolución de las figuras.")
    args = ap.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_table(infile)

    # Detecta columna de temperatura si no se indicó
    temp_col = args.temp_col
    if temp_col is None:
        temp_col = guess_column(df, ["temperature_K", "temp", "temperatura", "t_c", "temp_c"])
    if temp_col is None:
        raise SystemExit("No encuentro columna de temperatura. Pásala con --temp-col, p.ej. --temp-col temperature_c")

    # Detecta columna split si no existe (opcional; si no está, seguimos sin split)
    split_col = args.split_col if args.split_col in df.columns else None

    # Normaliza temperatura -> float (redondeada a 'args.round')
    temp_raw = df[temp_col].apply(lambda x: normalize_temperature(x, None))
    temp_c = temp_raw - 273.15
    temp_norm = temp_c.apply(lambda x: normalize_temperature(x, args.round))
    df = df.assign(_temp=temp_norm)
    df = df.dropna(subset=["_temp"])

    # Construye tabla de recuentos (por temperatura y split si existe)
    if split_col:
        grp = df.groupby(["_temp", split_col], dropna=False).size().rename("count").reset_index()
        pivot = grp.pivot(index="_temp", columns=split_col, values="count").fillna(0).astype(int)
        pivot["TOTAL"] = pivot.sum(axis=1)
        pivot = pivot.sort_index()
    else:
        pivot = df.groupby("_temp").size().rename("TOTAL").to_frame().sort_index()

    # Guarda CSV de recuentos
    counts_csv = outdir / "temperatura_counts.csv"
    pivot.to_csv(counts_csv)
    print(f"[OK] Recuentos guardados en: {counts_csv}")

    # Filtrado visual por min_count
    plot_pivot = pivot[pivot["TOTAL"] >= args.min_count].copy() if "TOTAL" in pivot.columns else pivot.copy()
    temps = plot_pivot.index.astype(float)

    # ===== Figura 1: barras TOTAL por temperatura =====
    plt.figure(figsize=(10, 5))
    if "TOTAL" in plot_pivot.columns:
        heights = plot_pivot["TOTAL"].values
    else:
        heights = plot_pivot.iloc[:, 0].values
    plt.bar(temps, heights, width=args.round if args.round else 0.8)
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Número de compuestos")
    plt.title("Compuestos por temperatura (TOTAL)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(temps, rotation=0)
    out1 = outdir / "hist_temperatura_overall.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=args.dpi)
    plt.close()
    print(f"[OK] Figura total: {out1}")

    # ===== Figura 2: barras agrupadas por split =====
    if split_col:
        cols = [c for c in plot_pivot.columns if c != "TOTAL"]
        # si hay muchos splits raros, ordena por total y limita a 3-4 principales (opcional)
        # aquí lo dejamos todo
        n_splits = len(cols)
        # ancho de barra por grupo
        group_width = 0.8
        bar_width = group_width / max(n_splits, 1)

        plt.figure(figsize=(11, 5.5))
        x = np.arange(len(temps), dtype=float)
        for i, c in enumerate(cols):
            plt.bar(x + (i - (n_splits - 1) / 2) * bar_width, plot_pivot[c].values,
                    width=bar_width, label=str(c))

        plt.xlabel("Temperatura (°C)")
        plt.ylabel("Número de compuestos")
        plt.title("Compuestos por temperatura y split")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(x, [f"{t:g}" for t in temps], rotation=0)
        plt.legend(title=split_col)
        out2 = outdir / "hist_temperatura_por_split.png"
        plt.tight_layout()
        plt.savefig(out2, dpi=args.dpi)
        plt.close()
        print(f"[OK] Figura por split: {out2}")

    # Mensaje de ayuda rápida
    print("\nSugerencias:")
    print("- Si tus temperaturas son discretas (p. ej. 20, 25, 30), usa --round 5.0.")
    print("- Si el split no está en una columna llamada 'split', pásala con --split-col.")
    print("- Revisa 'temperatura_counts.csv' para ver exactamente la distribución.")


if __name__ == "__main__":
    main()
