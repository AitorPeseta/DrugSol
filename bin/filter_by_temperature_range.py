#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
    filter_by_temperature_range.py
========================================================================================
    Filters solubility data to a specified temperature range.
    
    Temperature filtering is critical for solubility modeling because:
    - Solubility is strongly temperature-dependent (Van't Hoff equation)
    - Most drug applications target physiological temperatures
    - Extreme temperatures may introduce measurement artifacts
    - Consistent temperature range reduces model complexity
    
    Default Range: 24-50°C (configurable via --min and --max)
    - Lower bound (24°C): Slightly below room temperature
    - Upper bound (50°C): Above body temperature, captures fever conditions
    
    Features:
    - Inclusive or strict inequality modes
    - Optional preservation of NaN values
    - Supports both CSV and Parquet formats
    - Detailed logging with filter statistics

    Arguments:
        --input: Path to input dataset (CSV or Parquet)
        --out: Path to output filtered dataset (CSV or Parquet)
        --temp-col: Name of the temperature column (default: temp_C)
        --min: Minimum temperature (inclusive by default)
        --max: Maximum temperature (inclusive by default)
        --strict: Use strict inequality (min < x < max)
        --keep-na: Preserve rows with NaN temperature values
    
    Usage:
        python filter_by_temperature_range.py --input data.parquet --out filtered.parquet --min 24 --max 50

    Output:
        Filtered dataset saved to specified output path.
----------------------------------------------------------------------------------------
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# ======================================================================================
#     FILE I/O UTILITIES
# ======================================================================================

def read_any(path: str) -> pd.DataFrame:
    """
    Read CSV or Parquet file based on extension.
    
    Args:
        path: Path to input file
        
    Returns:
        DataFrame with file contents
    """
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_any(df: pd.DataFrame, path: str) -> None:
    """
    Write DataFrame to CSV or Parquet based on extension.
    
    Args:
        df: DataFrame to write
        path: Output file path
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    else:
        df.to_parquet(path, index=False)

# ======================================================================================
#     ARGUMENT PARSING
# ======================================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter rows by temperature range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter to 24-50°C range (default for drug solubility)
    python filter_by_temperature_range.py --input data.parquet --out filtered.parquet --min 24 --max 50
    
    # Filter to physiological range only
    python filter_by_temperature_range.py --input data.parquet --out filtered.parquet --min 35 --max 40
    
    # Keep rows with missing temperature
    python filter_by_temperature_range.py --input data.parquet --out filtered.parquet --min 24 --max 50 --keep-na
        """
    )
    parser.add_argument(
        "--input", 
        required=True,
        help="Input file (.parquet or .csv)"
    )
    parser.add_argument(
        "--out", 
        required=True,
        help="Output file (.parquet or .csv)"
    )
    parser.add_argument(
        "--temp-col", 
        default="temp_C",
        help="Name of the temperature column (default: temp_C)"
    )
    parser.add_argument(
        "--min", 
        type=float, 
        default=None,
        help="Minimum temperature (inclusive). Default: -inf"
    )
    parser.add_argument(
        "--max", 
        type=float, 
        default=None,
        help="Maximum temperature (inclusive). Default: +inf"
    )
    parser.add_argument(
        "--strict", 
        action="store_true",
        help="Use strict inequality (min < x < max) instead of inclusive"
    )
    parser.add_argument(
        "--keep-na", 
        action="store_true",
        help="Keep rows with NaN in the temperature column"
    )
    return parser.parse_args()

# ======================================================================================
#     FILTERING LOGIC
# ======================================================================================

def create_temperature_mask(
    temperatures: pd.Series,
    min_temp: float,
    max_temp: float,
    strict: bool = False,
    keep_na: bool = False
) -> Tuple[pd.Series, int]:
    """
    Create boolean mask for temperature filtering.
    
    Args:
        temperatures: Series of temperature values
        min_temp: Minimum temperature bound
        max_temp: Maximum temperature bound
        strict: If True, use strict inequality (exclusive bounds)
        keep_na: If True, preserve rows with NaN temperatures
        
    Returns:
        Tuple of (boolean mask, count of NaN values)
    """
    # Convert to numeric, coercing errors to NaN
    temp_numeric = pd.to_numeric(temperatures, errors="coerce")
    mask_valid = temp_numeric.notna()
    n_nan = int((~mask_valid).sum())
    
    # Apply range filter
    if strict:
        in_range = (temp_numeric > min_temp) & (temp_numeric < max_temp)
    else:
        in_range = (temp_numeric >= min_temp) & (temp_numeric <= max_temp)
    
    # Combine with NaN handling
    if keep_na:
        # Keep if in range OR if originally NaN
        final_mask = in_range | (~mask_valid)
    else:
        # Keep only if valid AND in range
        final_mask = mask_valid & in_range
    
    return final_mask, n_nan

# ======================================================================================
#     MAIN ENTRY POINT
# ======================================================================================

def main() -> None:
    """Main entry point for temperature filtering."""
    args = parse_args()

    # -------------------------------------------------------------------------
    # Load input data
    # -------------------------------------------------------------------------
    print(f"[Filter Temp] Loading: {args.input}")
    
    try:
        df = read_any(args.input)
    except Exception as e:
        sys.exit(f"[ERROR] Could not read file: {e}")

    if args.temp_col not in df.columns:
        sys.exit(f"[ERROR] Column '{args.temp_col}' not found in dataset.")

    initial_rows = len(df)

    # -------------------------------------------------------------------------
    # Define temperature bounds
    # -------------------------------------------------------------------------
    min_temp = -np.inf if args.min is None else float(args.min)
    max_temp = np.inf if args.max is None else float(args.max)

    # -------------------------------------------------------------------------
    # Apply filter
    # -------------------------------------------------------------------------
    mask, n_nan = create_temperature_mask(
        df[args.temp_col],
        min_temp,
        max_temp,
        strict=args.strict,
        keep_na=args.keep_na
    )
    
    df_filtered = df.loc[mask].copy()
    
    # -------------------------------------------------------------------------
    # Calculate statistics
    # -------------------------------------------------------------------------
    kept = len(df_filtered)
    dropped = initial_rows - kept
    
    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    write_any(df_filtered, args.out)

    # -------------------------------------------------------------------------
    # Report results
    # -------------------------------------------------------------------------
    operator_min = '>' if args.strict else '>='
    operator_max = '<' if args.strict else '<='
    
    print(f"[Filter Temp] Rows before:  {initial_rows:,}")
    print(f"[Filter Temp] Rows after:   {kept:,}")
    print(f"[Filter Temp] Dropped:      {dropped:,} ({100*dropped/initial_rows:.1f}%)")
    print(f"[Filter Temp] Range: {args.temp_col} {operator_min} {min_temp}°C AND {operator_max} {max_temp}°C")

    if not args.keep_na and n_nan > 0:
        print(f"[WARNING] Dropped {n_nan:,} rows with NaN/non-numeric temperature", file=sys.stderr)

    print(f"[Filter Temp] Saved: {args.out}")


if __name__ == "__main__":
    main()
