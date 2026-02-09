#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balance Dataset: Temperature-Aware Undersampling
=================================================

Performs intelligent dataset balancing to reduce temperature bias in solubility
data. Literature data is heavily skewed toward room temperature (25°C), but
physiological temperature (37°C) is more relevant for drug applications.

Balancing Strategy:
    The algorithm operates on 2D bins (logS × temperature):
    
    - Ambient (20-25°C): Apply undersampling limit per bin
    - Normal (25-35°C): Keep all samples
    - Physiological (35-40°C): Keep all samples (most valuable)
    - Hot (40-50°C): Keep all samples
    
    This preserves the solubility distribution while reducing the dominance
    of room temperature measurements.

Arguments:
    --input     : Input Parquet file with solubility data
    --output    : Output Parquet file path
    --limit     : Maximum samples per logS bin for ambient temperature (default: 50)
    --bin-size  : Size of logS bins in log units (default: 0.2)
    --seed      : Random seed for reproducible undersampling (default: 42)

Usage:
    python balance_dataset.py \\
        --input curated_data.parquet \\
        --output balanced_data.parquet \\
        --limit 25 \\
        --bin-size 0.2 \\
        --seed 42

Output:
    Parquet file with balanced temperature distribution.
    Ambient temperature samples are reduced while physiological data is preserved.

Notes:
    - Missing temperature values are filled with 25.0°C (room temperature assumption)
    - Smaller bin sizes provide finer control but may create sparse bins
    - The limit parameter controls the aggressiveness of undersampling
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# BALANCING FUNCTION
# ============================================================================

def balance_data_smart_2d(
    df: pd.DataFrame,
    log_col: str = 'logS',
    temp_col: str = 'temp_C',
    limit_ambient: int = 50,
    bin_size: float = 0.2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Balance dataset by undersampling ambient temperature data.
    
    The function creates 2D bins (solubility × temperature) and applies
    selective undersampling only to the ambient temperature range,
    preserving valuable physiological temperature data.
    
    Args:
        df: Input DataFrame with solubility and temperature columns
        log_col: Column name for log solubility (default: logS)
        temp_col: Column name for temperature in Celsius (default: temp_C)
        limit_ambient: Maximum samples per bin for ambient data (default: 50)
        bin_size: Width of logS bins in log units (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Balanced DataFrame with reduced ambient temperature samples
    """
    print(f"[Balance] Starting smart 2D balancing")
    print(f"          - LogS bin size: {bin_size}")
    print(f"          - Ambient (20-25°C) limit: {limit_ambient} per bin")
    print(f"          - Other temperatures: Keep all samples")

    # -------------------------------------------------------------------------
    # Create Solubility Bins
    # -------------------------------------------------------------------------
    min_log = np.floor(df[log_col].min())
    max_log = np.ceil(df[log_col].max())
    bins_log = np.arange(min_log, max_log + bin_size, bin_size)
    
    # -------------------------------------------------------------------------
    # Create Temperature Bins
    # -------------------------------------------------------------------------
    # Categories designed around physiologically relevant ranges
    bins_temp = [20, 25, 35, 40, 50]
    labels_temp = ['Ambient', 'Normal', 'Physio', 'Hot_Physio']
    
    # Assign bins
    df = df.copy()
    df['_bin_log'] = pd.cut(df[log_col], bins=bins_log, include_lowest=True)
    df['_bin_temp'] = pd.cut(df[temp_col], bins=bins_temp, labels=labels_temp, include_lowest=True)
    
    # -------------------------------------------------------------------------
    # Conditional Sampling Function
    # -------------------------------------------------------------------------
    def sampler(group):
        """Apply undersampling only to ambient temperature groups."""
        temp_label = group.name[1]  # Second element of MultiIndex is temp bin
        
        if temp_label == 'Ambient':
            n = len(group)
            if n > limit_ambient:
                return group.sample(limit_ambient, random_state=seed)
        return group
    
    # -------------------------------------------------------------------------
    # Execute Grouped Sampling
    # -------------------------------------------------------------------------
    df_balanced = df.groupby(
        ['_bin_log', '_bin_temp'],
        observed=True,
        group_keys=False
    ).apply(sampler)
    
    # -------------------------------------------------------------------------
    # Report Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f" BALANCE REPORT (Bin Size: {bin_size})")
    print("=" * 60)
    
    orig_counts = df['_bin_temp'].value_counts()
    new_counts = df_balanced['_bin_temp'].value_counts()
    
    for label in labels_temp:
        if label in orig_counts.index:
            orig = orig_counts[label]
            new = new_counts.get(label, 0)
            status = "Undersampled" if label == 'Ambient' else "Preserved"
            reduction = (1 - new / orig) * 100 if orig > 0 else 0
            print(f"  {label:12s}: {orig:6d} -> {new:6d}  [{status}, -{reduction:.1f}%]")
    
    print("-" * 60)
    total_orig = len(df)
    total_new = len(df_balanced)
    print(f"  Total: {total_orig:,} -> {total_new:,} ({total_new/total_orig*100:.1f}% retained)")
    
    # Remove temporary columns
    return df_balanced.drop(columns=['_bin_log', '_bin_temp'])


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for dataset balancing script."""
    
    ap = argparse.ArgumentParser(
        description="Balance solubility dataset by temperature-aware undersampling."
    )
    ap.add_argument("--input", required=True,
                    help="Input Parquet file")
    ap.add_argument("--output", required=True,
                    help="Output Parquet file")
    ap.add_argument("--limit", type=int, default=50,
                    help="Max samples per bin for ambient temperature (default: 50)")
    ap.add_argument("--bin-size", type=float, default=0.2,
                    help="LogS bin size in log units (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    if not Path(args.input).exists():
        sys.exit(f"[ERROR] Input file not found: {args.input}")
    
    print(f"[Balance] Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"[Balance] Loaded {len(df):,} rows")
    
    # -------------------------------------------------------------------------
    # Handle Missing Temperatures
    # -------------------------------------------------------------------------
    if df['temp_C'].isnull().any():
        n_missing = df['temp_C'].isnull().sum()
        print(f"[Balance] Filling {n_missing:,} missing temperatures with 25.0°C")
        df['temp_C'] = df['temp_C'].fillna(25.0)
    
    # -------------------------------------------------------------------------
    # Balance Dataset
    # -------------------------------------------------------------------------
    df_balanced = balance_data_smart_2d(
        df,
        limit_ambient=args.limit,
        bin_size=args.bin_size,
        seed=args.seed
    )
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    print(f"\n[Balance] Saving to {args.output}...")
    df_balanced.to_parquet(args.output, index=False)
    print("[Balance] Done.")


if __name__ == "__main__":
    main()
