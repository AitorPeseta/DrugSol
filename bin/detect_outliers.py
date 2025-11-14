#!/usr/bin/env python3
# detect_outliers.py
# Marca outliers de logS condicionados por temperatura mediante z-score por bins.
# Salida: MISMO esquema que la entrada + is_outlier (0/1). Nada más.

import argparse, os, sys
import numpy as np
import pandas as pd

MAD_SCALE = 1.4826  # para convertir MAD a sigma (asumiendo normal)

def make_bins(series: pd.Series, mode: str, n_bins: int | None, width: float | None):
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        raise SystemExit("[ERROR] La columna de temperatura no tiene valores válidos.")
    lo, hi = float(s.min()), float(s.max())
    if mode == "width":
        if width is None and n_bins is None:
            raise SystemExit("[ERROR] Con binning=width indica --bin-width o --bins.")
        if width is not None:
            start = np.floor(lo / width) * width
            stop = np.ceil(hi / width) * width + width
            edges = np.arange(start, stop, width, dtype=float)
        else:
            edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=float)
    elif mode == "quantile":
        n_bins = 10 if n_bins is None else int(n_bins)
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(s.quantile(qs).values.astype(float))
        if len(edges) < 2:
            edges = np.array([lo, hi], dtype=float)
    else:
        raise SystemExit("[ERROR] binning debe ser 'width' o 'quantile'.")
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([lo, hi], dtype=float)
    return edges

def zscore_per_bin(values: pd.Series, bins: pd.Categorical, method: str, min_count: int):
    """Devuelve dos Series alineadas a 'values': z-score y mascara de escala cero."""
    z = pd.Series(0.0, index=values.index, dtype=float)
    scale_zero = pd.Series(False, index=values.index, dtype=bool)

    # groupby sobre los labels de bins (categorical) – observed=True para silenciar FutureWarning
    for b, idx in bins.groupby(bins, observed=True).groups.items():
        vs = values.loc[idx].astype(float)
        if vs.size < min_count:
            scale_zero.loc[idx] = True
            z.loc[idx] = 0.0
            continue
        if method == "robust":
            med = float(vs.median())
            mad = float((vs - med).abs().median())
            scale = MAD_SCALE * mad
            if not np.isfinite(scale) or scale == 0:
                scale_zero.loc[idx] = True
                z.loc[idx] = 0.0
            else:
                z.loc[idx] = (vs - med) / scale
        else:
            mu = float(vs.mean())
            sd = float(vs.std(ddof=1))
            if not np.isfinite(sd) or sd == 0:
                scale_zero.loc[idx] = True
                z.loc[idx] = 0.0
            else:
                z.loc[idx] = (vs - mu) / sd
    return z, scale_zero

def main():
    ap = argparse.ArgumentParser(description="Detecta outliers de logS condicionados por temperatura (bins + z-score).")
    ap.add_argument("--input", required=True, help="Parquet de entrada")
    ap.add_argument("--out", default="detect_outliers.parquet", help="Parquet de salida")
    ap.add_argument("--export-csv", action="store_true", help="Exporta también CSV")
    ap.add_argument("--log-col", default="logS", help="Columna de solubilidad (default: logS)")
    ap.add_argument("--temp-col", default="temperature_K", help="Columna de temperatura (default: temperature_K)")
    ap.add_argument("--binning", choices=["width", "quantile"], default="width",
                    help="Estrategia de binning: width (anchura fija o nº de bins iguales) o quantile")
    ap.add_argument("--bin-width", type=float, default=None, help="Anchura del bin si binning=width (ignorado si --bins)")
    ap.add_argument("--bins", type=int, default=10, help="Nº de bins (width: equiespaciados; quantile: equipoblados)")
    ap.add_argument("--z-method", choices=["standard", "robust"], default="robust",
                    help="z-score estándar (media/SD) o robusto (mediana/MAD)")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="Umbral de outlier: |z| > z-thresh")
    ap.add_argument("--min-count", type=int, default=8, help="Mínimo de puntos por bin para calcular z")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    for c in (args.log_col, args.temp_col):
        if c not in df.columns:
            raise SystemExit(f"[ERROR] La columna '{c}' no existe en el dataset.")

    # Validos numéricamente en ambas columnas
    s_log = pd.to_numeric(df[args.log_col], errors="coerce")
    s_tmp = pd.to_numeric(df[args.temp_col], errors="coerce")
    valid = s_log.notna() & s_tmp.notna()

    if (~valid).any():
        n_drop = int((~valid).sum())
        print(f"[AVISO] Filas inválidas (NaN/no numéricas) en {args.log_col}/{args.temp_col}: {n_drop} (no se consideran)", file=sys.stderr)

    # Construye bins sobre temperaturas válidas
    edges = make_bins(s_tmp[valid], mode=args.binning, n_bins=args.bins, width=args.bin_width)
    # Categorización (NO se guarda en la salida)
    temp_bins = pd.cut(s_tmp[valid].astype(float), bins=edges, include_lowest=True)

    # z-score por bin
    z, scale_zero = zscore_per_bin(s_log[valid].astype(float), temp_bins, args.z_method, args.min_count)

    # Regla de outlier: |z| > z_thresh y bin con escala válida
    is_out = ((z.abs() > args.z_thresh) & (~scale_zero)).astype(int)

    # Salida: MISMO DF + is_outlier (el resto intacto)
    df_out = df.copy()
    df_out["is_outlier"] = 0
    df_out.loc[valid.index, "is_outlier"] = is_out.reindex(valid.index).fillna(0).astype(int)

    # Guarda parquet (sin columnas extra que puedan romper a pyarrow)
    df_out.to_parquet(args.out)
    if args.export_csv:
        csv_path = os.path.splitext(args.out)[0] + ".csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")

    # Resumen
    total = len(df_out)
    n_out = int(df_out["is_outlier"].sum())
    pct = (n_out / total * 100) if total else 0.0
    print(f"[detect_outliers] OK  total={total}  outliers={n_out} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
