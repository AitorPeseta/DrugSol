#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
    filter_water.py
========================================================================================
    Filters solubility dataset to retain only aqueous measurements.
    
    Aqueous solubility is the primary focus of drug solubility prediction since:
    - Most biological systems are aqueous environments
    - Drug absorption typically occurs in aqueous media (GI tract, blood)
    - Regulatory guidelines (FDA, EMA) focus on aqueous solubility
    - Thermodynamic models are most accurate for aqueous systems
    
    This filter removes measurements in:
    - Organic solvents (ethanol, methanol, DMSO, etc.)
    - Mixed solvent systems (water-ethanol mixtures)
    - Unknown or unspecified solvents
    
    Filter Logic:
    - Case-insensitive exact match for 'water'
    - Whitespace-tolerant comparison
    - Preserves all other columns unchanged

    Arguments:
        --input     Input Parquet file path
        --output    Output Parquet file path
    
    Usage:
        python filter_water.py --input unified.parquet --output filtered.parquet

    Output:
        Parquet file with only aqueous solubility entries.
----------------------------------------------------------------------------------------
"""

import argparse
import sys

import pandas as pd

# ======================================================================================
#     CONSTANTS
# ======================================================================================

# Target solvent for filtering (case-insensitive)
TARGET_SOLVENT = "water"

# ======================================================================================
#     ARGUMENT PARSING
# ======================================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter dataset to keep only aqueous solubility entries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python filter_water.py --input unified.parquet --output filter_water.parquet
        """
    )
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to input Parquet file"
    )
    parser.add_argument(
        "--output", 
        required=True,
        help="Path to output Parquet file"
    )
    return parser.parse_args()

# ======================================================================================
#     FILTERING LOGIC
# ======================================================================================

def filter_aqueous(df: pd.DataFrame, solvent_col: str = "solvent") -> pd.DataFrame:
    """
    Filter DataFrame to retain only aqueous solubility measurements.
    
    Args:
        df: Input DataFrame with solvent column
        solvent_col: Name of the solvent column
        
    Returns:
        Filtered DataFrame with only water-based entries
    """
    # Normalize solvent values: strip whitespace, convert to lowercase
    solvent_normalized = df[solvent_col].astype(str).str.strip().str.lower()
    
    # Filter for exact match with 'water'
    mask = solvent_normalized == TARGET_SOLVENT
    
    return df[mask].copy()

# ======================================================================================
#     MAIN ENTRY POINT
# ======================================================================================

def main() -> None:
    """Main entry point for water filtering."""
    args = parse_args()

    # -------------------------------------------------------------------------
    # Load input data
    # -------------------------------------------------------------------------
    print(f"[Filter Water] Loading: {args.input}")
    
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to read input file: {e}")
        sys.exit(1)

    initial_rows = len(df)

    # -------------------------------------------------------------------------
    # Apply filter
    # -------------------------------------------------------------------------
    df_water = filter_aqueous(df)
    final_rows = len(df_water)
    dropped = initial_rows - final_rows

    # -------------------------------------------------------------------------
    # Report statistics
    # -------------------------------------------------------------------------
    print(f"[Filter Water] Rows before:  {initial_rows:,}")
    print(f"[Filter Water] Rows after:   {final_rows:,}")
    print(f"[Filter Water] Dropped:      {dropped:,} ({100*dropped/initial_rows:.1f}%)")

    if final_rows == 0:
        print("[WARNING] Resulting dataset is empty!")
        print("         Check if 'solvent' column contains 'water' values.")
        
        # Show unique solvent values for debugging
        unique_solvents = df['solvent'].astype(str).str.strip().str.lower().unique()[:10]
        print(f"         Sample solvent values: {list(unique_solvents)}")

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    df_water.to_parquet(args.output, index=False)
    print(f"[Filter Water] Saved: {args.output}")


if __name__ == "__main__":
    main()
