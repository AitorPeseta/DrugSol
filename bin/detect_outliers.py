#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
    detect_outliers.py
========================================================================================
    Detects statistical outliers in solubility data using temperature-conditioned
    robust z-scores.
    
    Why Temperature-Conditioned?
    - Solubility varies significantly with temperature (Van't Hoff relationship)
    - A "normal" logS at 25°C might be an outlier at 50°C
    - Binning by temperature ensures fair comparison within similar conditions
    
    Method: Robust Z-Score (Median Absolute Deviation)
    - More resistant to outliers than standard z-score (mean/std)
    - Formula: Z = (x - median) / (MAD × 1.4826)
    - The constant 1.4826 makes MAD consistent with std for normal distributions
    - Threshold: |Z| > 3.0 flagged as outlier (configurable)
    
    Algorithm:
    1. Bin data by temperature (equal-width or quantile bins)
    2. Calculate robust z-score within each temperature bin
    3. Flag points where |z| > threshold
    4. Preserve original schema, add is_outlier column
    
    Arguments:
        --input       Input Parquet file with logS and temperature data
        --out         Output Parquet file (default: detect_outliers.parquet)
        --export-csv  Also export CSV copy
        --log-col     Solubility column name (default: logS)
        --temp-col    Temperature column name (default: temp_C)
        --binning     Binning strategy: 'width' or 'quantile' (default: width)
        --bin-width   Bin width in °C (overrides --bins)
        --bins        Number of bins (default: 10)
        --z-method    Z-score method: 'standard' or 'robust' (default: robust)
        --z-thresh    Z-score threshold for outlier flagging (default: 3.0)
        --min-count   Minimum samples per bin to calculate stats (default: 8)

    Usage:
        python detect_outliers.py --input filtered.parquet --out outliers.parquet
    
    Output:
        Parquet file with original columns + 'is_outlier' (0 = normal, 1 = outlier)
----------------------------------------------------------------------------------------
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ======================================================================================
#     CONSTANTS
# ======================================================================================

# Scale factor to convert MAD to standard deviation equivalent
# For normal distribution: std ≈ MAD × 1.4826
MAD_TO_STD_SCALE = 1.4826

# Default parameters
DEFAULT_BINS = 10
DEFAULT_Z_THRESHOLD = 3.0
DEFAULT_MIN_COUNT = 8

# ======================================================================================
#     ARGUMENT PARSING
# ======================================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect conditional outliers in logS vs Temperature.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with defaults (robust z-score, 10 bins, threshold 3.0)
    python detect_outliers.py --input filtered.parquet --out outliers.parquet
    
    # Use quantile binning with stricter threshold
    python detect_outliers.py --input data.parquet --out outliers.parquet \\
        --binning quantile --z-thresh 2.5
    
    # Fixed bin width of 5°C
    python detect_outliers.py --input data.parquet --out outliers.parquet \\
        --bin-width 5.0
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input", 
        required=True,
        help="Input Parquet file"
    )
    parser.add_argument(
        "--out", 
        default="detect_outliers.parquet",
        help="Output Parquet file (default: detect_outliers.parquet)"
    )
    parser.add_argument(
        "--export-csv", 
        action="store_true",
        help="Also export a CSV copy"
    )
    
    # Column names
    parser.add_argument(
        "--log-col", 
        default="logS",
        help="Solubility column name (default: logS)"
    )
    parser.add_argument(
        "--temp-col", 
        default="temp_C",
        help="Temperature column name (default: temp_C)"
    )
    
    # Binning options
    parser.add_argument(
        "--binning", 
        choices=["width", "quantile"], 
        default="width",
        help="Binning strategy: 'width' (equal width) or 'quantile' (equal population)"
    )
    parser.add_argument(
        "--bin-width", 
        type=float, 
        default=None,
        help="Bin width in °C (overrides --bins)"
    )
    parser.add_argument(
        "--bins", 
        type=int, 
        default=DEFAULT_BINS,
        help=f"Number of bins (default: {DEFAULT_BINS})"
    )
    
    # Z-score options
    parser.add_argument(
        "--z-method", 
        choices=["standard", "robust"], 
        default="robust",
        help="Method: 'standard' (mean/std) or 'robust' (median/MAD)"
    )
    parser.add_argument(
        "--z-thresh", 
        type=float, 
        default=DEFAULT_Z_THRESHOLD,
        help=f"Z-score threshold to flag outlier (default: {DEFAULT_Z_THRESHOLD})"
    )
    parser.add_argument(
        "--min-count", 
        type=int, 
        default=DEFAULT_MIN_COUNT,
        help=f"Minimum samples per bin to calculate stats (default: {DEFAULT_MIN_COUNT})"
    )
    
    return parser.parse_args()

# ======================================================================================
#     BINNING FUNCTIONS
# ======================================================================================

def create_bin_edges(
    series: pd.Series,
    mode: str,
    n_bins: Optional[int],
    width: Optional[float]
) -> np.ndarray:
    """
    Generate bin edges for temperature binning.
    
    Args:
        series: Temperature values
        mode: 'width' for equal-width bins, 'quantile' for equal-population
        n_bins: Number of bins (used if width not specified)
        width: Bin width in temperature units (overrides n_bins)
        
    Returns:
        Array of bin edges
    """
    # Clean and convert to numeric
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    
    if values.empty:
        sys.exit("[ERROR] Temperature column has no valid values.")
    
    lo, hi = float(values.min()), float(values.max())
    
    if mode == "width":
        if width is not None:
            # Fixed width bins
            start = np.floor(lo / width) * width
            stop = np.ceil(hi / width) * width + width
            edges = np.arange(start, stop, width, dtype=float)
        elif n_bins is not None:
            # Fixed number of equal-width bins
            edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=float)
        else:
            sys.exit("[ERROR] For binning=width, provide --bin-width or --bins.")
            
    elif mode == "quantile":
        # Equal-population bins based on quantiles
        n_bins = DEFAULT_BINS if n_bins is None else int(n_bins)
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(values.quantile(quantiles).values.astype(float))
        
        # Fallback if quantiles collapse
        if len(edges) < 2:
            edges = np.array([lo, hi], dtype=float)
    else:
        sys.exit(f"[ERROR] Unknown binning mode: {mode}")
    
    # Ensure unique edges
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([lo, hi], dtype=float)
    
    return edges

# ======================================================================================
#     Z-SCORE CALCULATION
# ======================================================================================

def calculate_zscore_per_bin(
    values: pd.Series,
    bins: pd.Categorical,
    method: str,
    min_count: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate z-scores for each value relative to its bin statistics.
    
    Args:
        values: Solubility values (logS)
        bins: Categorical series of bin assignments
        method: 'standard' (mean/std) or 'robust' (median/MAD)
        min_count: Minimum samples required per bin
        
    Returns:
        Tuple of (z-scores, scale_zero_mask)
        - z-scores: Calculated z-score for each point
        - scale_zero_mask: Boolean mask where variance was zero/undefined
    """
    z_scores = pd.Series(0.0, index=values.index, dtype=float)
    scale_zero = pd.Series(False, index=values.index, dtype=bool)
    
    # Group by bin labels
    groups = bins.groupby(bins, observed=True).groups
    
    for bin_label, indices in groups.items():
        bin_values = values.loc[indices].astype(float)
        
        # Skip bins with insufficient data
        if bin_values.size < min_count:
            scale_zero.loc[indices] = True
            z_scores.loc[indices] = 0.0
            continue
        
        if method == "robust":
            # Robust Z-Score: (x - Median) / (MAD × 1.4826)
            median = float(bin_values.median())
            mad = float((bin_values - median).abs().median())
            scale = MAD_TO_STD_SCALE * mad
            
            if not np.isfinite(scale) or scale == 0:
                scale_zero.loc[indices] = True
                z_scores.loc[indices] = 0.0
            else:
                z_scores.loc[indices] = (bin_values - median) / scale
        else:
            # Standard Z-Score: (x - Mean) / StdDev
            mean = float(bin_values.mean())
            std = float(bin_values.std(ddof=1))
            
            if not np.isfinite(std) or std == 0:
                scale_zero.loc[indices] = True
                z_scores.loc[indices] = 0.0
            else:
                z_scores.loc[indices] = (bin_values - mean) / std
    
    return z_scores, scale_zero

# ======================================================================================
#     MAIN ENTRY POINT
# ======================================================================================

def main() -> None:
    """Main entry point for outlier detection."""
    args = parse_args()

    # -------------------------------------------------------------------------
    # Load input data
    # -------------------------------------------------------------------------
    print(f"[Detect Outliers] Loading: {args.input}")
    
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read input: {e}")

    # Validate required columns
    for col in (args.log_col, args.temp_col):
        if col not in df.columns:
            sys.exit(f"[ERROR] Required column '{col}' not found in dataset.")

    initial_rows = len(df)

    # -------------------------------------------------------------------------
    # Validate numeric data
    # -------------------------------------------------------------------------
    log_values = pd.to_numeric(df[args.log_col], errors="coerce")
    temp_values = pd.to_numeric(df[args.temp_col], errors="coerce")
    valid_mask = log_values.notna() & temp_values.notna()

    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        print(f"[WARNING] Ignoring {n_invalid:,} rows with NaN in {args.log_col} or {args.temp_col}", 
              file=sys.stderr)

    # -------------------------------------------------------------------------
    # Create temperature bins
    # -------------------------------------------------------------------------
    edges = create_bin_edges(
        temp_values[valid_mask],
        mode=args.binning,
        n_bins=args.bins,
        width=args.bin_width
    )
    
    temp_bins = pd.cut(
        temp_values[valid_mask].astype(float),
        bins=edges,
        include_lowest=True
    )
    
    print(f"[Detect Outliers] Temperature bins: {len(edges) - 1} ({args.binning} mode)")

    # -------------------------------------------------------------------------
    # Calculate z-scores
    # -------------------------------------------------------------------------
    z_scores, scale_zero = calculate_zscore_per_bin(
        log_values[valid_mask].astype(float),
        temp_bins,
        args.z_method,
        args.min_count
    )

    # -------------------------------------------------------------------------
    # Flag outliers
    # -------------------------------------------------------------------------
    # Rule: |z| > threshold AND scale was valid (not zero)
    is_outlier = ((z_scores.abs() > args.z_thresh) & (~scale_zero)).astype(int)

    # -------------------------------------------------------------------------
    # Prepare output
    # -------------------------------------------------------------------------
    df_out = df.copy()
    df_out["is_outlier"] = 0
    df_out.loc[valid_mask, "is_outlier"] = is_outlier.reindex(valid_mask.index).fillna(0).astype(int)

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    df_out.to_parquet(args.out, index=False)
    
    if args.export_csv:
        csv_path = os.path.splitext(args.out)[0] + ".csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
        print(f"[Detect Outliers] CSV exported: {csv_path}")

    # -------------------------------------------------------------------------
    # Report statistics
    # -------------------------------------------------------------------------
    n_outliers = int(df_out["is_outlier"].sum())
    pct = (n_outliers / initial_rows * 100) if initial_rows else 0.0
    
    print(f"[Detect Outliers] Total rows:  {initial_rows:,}")
    print(f"[Detect Outliers] Outliers:    {n_outliers:,} ({pct:.2f}%)")
    print(f"[Detect Outliers] Method:      {args.z_method} z-score, threshold={args.z_thresh}")
    print(f"[Detect Outliers] Saved: {args.out}")


if __name__ == "__main__":
    main()
