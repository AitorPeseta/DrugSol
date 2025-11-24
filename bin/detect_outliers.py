#!/usr/bin/env python3
# detect_outliers.py
# Flags logS outliers conditioned on temperature using binned z-scores.
# Output: Same schema as input + 'is_outlier' column (0/1).

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Constant to scale MAD to Sigma (assuming normal distribution consistency)
MAD_SCALE = 1.4826

def make_bins(series: pd.Series, mode: str, n_bins: int | None, width: float | None):
    """Generates bin edges based on the strategy (width or quantile)."""
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        sys.exit("[ERROR] Temperature column has no valid values.")
        
    lo, hi = float(s.min()), float(s.max())
    
    if mode == "width":
        if width is None and n_bins is None:
            sys.exit("[ERROR] For binning=width, you must provide --bin-width or --bins.")
            
        if width is not None:
            start = np.floor(lo / width) * width
            stop = np.ceil(hi / width) * width + width
            edges = np.arange(start, stop, width, dtype=float)
        else:
            # Fixed number of equal-width bins
            edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=float)
            
    elif mode == "quantile":
        n_bins = 10 if n_bins is None else int(n_bins)
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(s.quantile(qs).values.astype(float))
        if len(edges) < 2:
            edges = np.array([lo, hi], dtype=float)
            
    else:
        sys.exit("[ERROR] binning must be 'width' or 'quantile'.")
        
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([lo, hi], dtype=float)
        
    return edges

def zscore_per_bin(values: pd.Series, bins: pd.Categorical, method: str, min_count: int):
    """
    Calculates z-score for each point relative to its bin statistics.
    Returns:
        z (Series): The calculated z-scores.
        scale_zero (Series): Boolean mask where variance was zero (to avoid div/0).
    """
    z = pd.Series(0.0, index=values.index, dtype=float)
    scale_zero = pd.Series(False, index=values.index, dtype=bool)

    # Group by bin labels
    # observed=True silences pandas FutureWarnings for categorical grouping
    groups = bins.groupby(bins, observed=True).groups
    
    for b, idx in groups.items():
        vs = values.loc[idx].astype(float)
        
        # Skip bins with too few points
        if vs.size < min_count:
            scale_zero.loc[idx] = True
            z.loc[idx] = 0.0
            continue
            
        if method == "robust":
            # Robust Z-Score: (x - Median) / (MAD * 1.4826)
            med = float(vs.median())
            mad = float((vs - med).abs().median())
            scale = MAD_SCALE * mad
            
            if not np.isfinite(scale) or scale == 0:
                scale_zero.loc[idx] = True
                z.loc[idx] = 0.0
            else:
                z.loc[idx] = (vs - med) / scale
        else:
            # Standard Z-Score: (x - Mean) / StdDev
            mu = float(vs.mean())
            sd = float(vs.std(ddof=1))
            
            if not np.isfinite(sd) or sd == 0:
                scale_zero.loc[idx] = True
                z.loc[idx] = 0.0
            else:
                z.loc[idx] = (vs - mu) / sd
                
    return z, scale_zero

def main():
    ap = argparse.ArgumentParser(description="Detect conditional outliers in logS vs Temperature.")
    ap.add_argument("--input", required=True, help="Input parquet file")
    ap.add_argument("--out", default="detect_outliers.parquet", help="Output parquet file")
    ap.add_argument("--export-csv", action="store_true", help="Also export a CSV copy")
    ap.add_argument("--log-col", default="logS", help="Solubility column name (default: logS)")
    ap.add_argument("--temp-col", default="temp_C", help="Temperature column name (default: temp_C)")
    
    ap.add_argument("--binning", choices=["width", "quantile"], default="width",
                    help="Binning strategy: 'width' (equal width) or 'quantile' (equal population).")
    ap.add_argument("--bin-width", type=float, default=None, help="Bin width (if binning=width). Overrides --bins.")
    ap.add_argument("--bins", type=int, default=10, help="Number of bins (default: 10).")
    
    ap.add_argument("--z-method", choices=["standard", "robust"], default="robust",
                    help="Method: 'standard' (mean/sd) or 'robust' (median/mad).")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold to flag outlier (default: 3.0).")
    ap.add_argument("--min-count", type=int, default=8, help="Min samples per bin to calculate stats (default: 8).")
    
    args = ap.parse_args()

    # 1. Load Data
    print(f"[Detect Outliers] Loading {args.input} ...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read input: {e}")

    for c in (args.log_col, args.temp_col):
        if c not in df.columns:
            sys.exit(f"[ERROR] Required column '{c}' not found in dataset.")

    # 2. Validate Numeric Data
    s_log = pd.to_numeric(df[args.log_col], errors="coerce")
    s_tmp = pd.to_numeric(df[args.temp_col], errors="coerce")
    valid = s_log.notna() & s_tmp.notna()

    if (~valid).any():
        n_drop = int((~valid).sum())
        print(f"[WARNING] Ignoring {n_drop} rows with NaN in {args.log_col} or {args.temp_col}.", file=sys.stderr)

    # 3. Create Temperature Bins (only on valid data)
    edges = make_bins(s_tmp[valid], mode=args.binning, n_bins=args.bins, width=args.bin_width)
    
    # pd.cut returns a Categorical series
    temp_bins = pd.cut(s_tmp[valid].astype(float), bins=edges, include_lowest=True)

    # 4. Calculate Z-Scores
    z, scale_zero = zscore_per_bin(s_log[valid].astype(float), temp_bins, args.z_method, args.min_count)

    # 5. Flag Outliers
    # Rule: |z| > threshold AND scale was not zero
    is_out = ((z.abs() > args.z_thresh) & (~scale_zero)).astype(int)

    # 6. Prepare Output (Preserve original schema + is_outlier)
    df_out = df.copy()
    df_out["is_outlier"] = 0
    # Map calculated outliers back to original indices
    df_out.loc[valid.index, "is_outlier"] = is_out.reindex(valid.index).fillna(0).astype(int)

    # 7. Save
    df_out.to_parquet(args.out)
    if args.export_csv:
        csv_path = os.path.splitext(args.out)[0] + ".csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")

    # Summary
    total = len(df_out)
    n_out = int(df_out["is_outlier"].sum())
    pct = (n_out / total * 100) if total else 0.0
    print(f"[Detect Outliers] OK. Total: {total} | Outliers: {n_out} ({pct:.2f}%)")

if __name__ == "__main__":
    main()