#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
    unify_data_sets.py
========================================================================================
    Combines multiple solubility datasets into a unified standardized schema.
    
    This script handles the challenge of integrating data from diverse sources
    (BigSolDB, ChEMBL, Reaxys, Challenge datasets) that use different column
    names, units, and formats.
    
    Output Schema:
    - smiles_original: Original SMILES string from source
    - smiles_neutral:  Neutralized SMILES (if available)
    - solvent:         Solvent identifier (typically 'water')
    - temp_C:          Temperature in Celsius (auto-converted from Kelvin)
    - logS:            Log10 of molar solubility (mol/L)
    - is_outlier:      Outlier flag (populated in curation step)
    - source:          Data origin identifier for provenance tracking
    
    Features:
    - Automatic column name mapping from common variations
    - Temperature unit detection and conversion (K → C)
    - Robust file reading with encoding fallbacks
    - Source tracking for data provenance

    Arguments:
        --sources   List of input CSV/Parquet files to unify
        --export-csv Optional flag to export unified.csv in addition to Parquet
        --output    Base name for output files (default: unified)
    
    Usage:
        python unify_data_sets.py --sources file1.csv file2.parquet --export-csv
    
    Output:
        unified.parquet
----------------------------------------------------------------------------------------
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# ======================================================================================
#     SCHEMA CONFIGURATION
# ======================================================================================

# Columns required in the final unified output
REQUIRED_COLUMNS = [
    'smiles_original',
    'smiles_neutral',
    'solvent',
    'temp_C',
    'logS',
    'is_outlier',
    'source'
]

# Column name mapping: target -> list of possible source names
# The first match found will be used
COLUMN_CANDIDATES = {
    'smiles_original': [
        'smiles_original', 
        'canonical_smiles', 
        'SMILES', 
        'smiles', 
        'SMILES_Solute'
    ],
    'smiles_neutral': [
        'smiles_neutral'
    ],
    'solvent': [
        'solvent', 
        'Solvent', 
        'Medium', 
        'SMILES_Solvent', 
        'Solvent (Solubility (MCS))'
    ],
    # Temperature columns in Celsius (no conversion needed)
    'temp_C_sources': [
        'temp_C', 
        'Temperature_C', 
        'T_C', 
        'Temp_C'
    ],
    # Temperature columns in Kelvin (need -273.15 conversion)
    'temp_K_sources': [
        'temperature_K', 
        'Temperature_K', 
        'temp_K', 
        'T_K', 
        'T'
    ],
    'logS': [
        'logS', 
        'LogS', 
        'logS_molL', 
        'Solubility', 
        'Y', 
        'LogS(mol/L)'
    ],
    'is_outlier': [
        'is_outlier', 
        'outlier'
    ],
    'source': [
        'source', 
        'Source', 
        'dataset', 
        'origin'
    ],
}

# ======================================================================================
#     FILE READING UTILITIES
# ======================================================================================

def read_table(path: str) -> pd.DataFrame:
    """
    Read CSV or Parquet file with robust fallbacks.
    
    Handles common issues:
    - Different file formats (CSV, Parquet)
    - Encoding problems in CSV files
    - Alternative delimiters (tab-separated)
    
    Args:
        path: Path to input file
        
    Returns:
        DataFrame with file contents
    """
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    
    # Try standard CSV reading first
    try:
        return pd.read_csv(path)
    except Exception:
        pass
    
    # Fallback: try tab-separated with encoding error handling
    try:
        return pd.read_csv(path, sep='\t', engine='python', encoding_errors='ignore')
    except Exception:
        pass
    
    # Final fallback: Python engine with error handling
    return pd.read_csv(path, engine='python', encoding_errors='ignore')

# ======================================================================================
#     COLUMN MAPPING UTILITIES
# ======================================================================================

def pick_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Find and return the first matching column from candidate list.
    
    Args:
        df: Source DataFrame
        candidates: List of possible column names (in priority order)
        
    Returns:
        Series with column data, or NaN-filled Series if no match found
    """
    for col_name in candidates:
        if col_name in df.columns:
            return df[col_name]
    
    # Return empty series if no match found
    return pd.Series([np.nan] * len(df), index=df.index)


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors='coerce')


def to_boolean_safe(series: pd.Series) -> pd.Series:
    """
    Convert series to boolean with robust handling of various formats.
    
    Handles: 'yes', 'no', 'true', 'false', '1', '0', 'y', 'n', 't', 'f'
    
    Args:
        series: Input series with boolean-like values
        
    Returns:
        Nullable boolean series
    """
    if pd.api.types.is_bool_dtype(series):
        return series
    
    def convert_value(x):
        if pd.isna(x):
            return np.nan
        text = str(x).strip().lower()
        if text in ('true', '1', 'yes', 'y', 't'):
            return True
        if text in ('false', '0', 'no', 'n', 'f'):
            return False
        return np.nan
    
    return series.apply(convert_value).astype('boolean')

# ======================================================================================
#     DATASET LOADING AND STANDARDIZATION
# ======================================================================================

def load_and_standardize(path: str) -> pd.DataFrame:
    """
    Load a single dataset file and standardize to unified schema.
    
    Performs:
    1. Column name mapping from various source formats
    2. Temperature unit conversion (Kelvin → Celsius)
    3. Type enforcement for all columns
    4. Schema validation
    
    Args:
        path: Path to input file (CSV or Parquet)
        
    Returns:
        DataFrame with standardized schema
    """
    raw = read_table(path)
    out = pd.DataFrame(index=raw.index)

    # -------------------------------------------------------------------------
    # String Columns
    # -------------------------------------------------------------------------
    out['smiles_original'] = pick_series(raw, COLUMN_CANDIDATES['smiles_original']).astype('string')
    out['smiles_neutral'] = pick_series(raw, COLUMN_CANDIDATES['smiles_neutral']).astype('string')
    out['solvent'] = pick_series(raw, COLUMN_CANDIDATES['solvent']).astype('string')
    
    # Source column: use file name if not present in data
    source_series = pick_series(raw, COLUMN_CANDIDATES['source'])
    if source_series.isna().all():
        filename = os.path.basename(path)
        out['source'] = filename
    else:
        out['source'] = source_series.astype(str)

    # -------------------------------------------------------------------------
    # Temperature: Smart Celsius/Kelvin Handling
    # -------------------------------------------------------------------------
    # Try to find Celsius column first
    temp_celsius = to_numeric_safe(pick_series(raw, COLUMN_CANDIDATES['temp_C_sources']))
    
    # Try to find Kelvin column and convert
    temp_kelvin = to_numeric_safe(pick_series(raw, COLUMN_CANDIDATES['temp_K_sources']))
    temp_from_kelvin = temp_kelvin - 273.15
    
    # Merge: prefer Celsius if available, fall back to converted Kelvin
    out['temp_C'] = temp_celsius.fillna(temp_from_kelvin).round(2)

    # -------------------------------------------------------------------------
    # Target and Flags
    # -------------------------------------------------------------------------
    out['logS'] = to_numeric_safe(pick_series(raw, COLUMN_CANDIDATES['logS']))
    out['is_outlier'] = to_boolean_safe(pick_series(raw, COLUMN_CANDIDATES['is_outlier']))

    # -------------------------------------------------------------------------
    # Schema Validation: Ensure all required columns exist
    # -------------------------------------------------------------------------
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series([np.nan] * len(out), index=out.index)
    
    # Keep only required columns in correct order
    out = out[REQUIRED_COLUMNS]

    # -------------------------------------------------------------------------
    # Final Type Enforcement
    # -------------------------------------------------------------------------
    for col in ['smiles_original', 'smiles_neutral', 'solvent', 'source']:
        out[col] = out[col].astype('string')
    
    for col in ['temp_C', 'logS']:
        out[col] = to_numeric_safe(out[col])

    return out

# ======================================================================================
#     MAIN ENTRY POINT
# ======================================================================================

def main() -> None:
    """Main entry point for dataset unification."""
    parser = argparse.ArgumentParser(
        description="Unify multiple solubility datasets into a standard schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Unify multiple CSV files
    python unify_data_sets.py --sources bigsoldb.csv chembl.csv reaxys.csv
    
    # Include Parquet files and export CSV
    python unify_data_sets.py --sources *.csv *.parquet --export-csv
        """
    )
    parser.add_argument(
        '--sources', 
        nargs='+', 
        required=True,
        help='Input CSV/Parquet files to unify'
    )
    parser.add_argument(
        '--export-csv', 
        action='store_true',
        help='Export unified.csv in addition to Parquet'
    )
    parser.add_argument(
        '--output',
        default='unified',
        help='Output file base name (default: unified)'
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Process each source file
    # -------------------------------------------------------------------------
    dataframes = []
    
    for filepath in args.sources:
        try:
            print(f"[Unify] Processing: {filepath}")
            df = load_and_standardize(filepath)
            
            # Report statistics
            valid_temps = df['temp_C'].notna().sum()
            valid_logs = df['logS'].notna().sum()
            print(f"        Rows: {len(df):,} | Valid temps: {valid_temps:,} | Valid logS: {valid_logs:,}")
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"[Unify] WARNING: Failed to process {filepath}: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Handle empty case
    # -------------------------------------------------------------------------
    if not dataframes:
        print("[Unify] No valid data found. Creating empty dataset.")
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        empty_df.to_parquet(f'{args.output}.parquet', index=False)
        if args.export_csv:
            empty_df.to_csv(f'{args.output}.csv', index=False)
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Concatenate all sources
    # -------------------------------------------------------------------------
    unified = pd.concat(dataframes, axis=0, ignore_index=True)
    
    print("-" * 60)
    print(f"[Unify] Total rows after unification: {len(unified):,}")
    print(f"[Unify] Columns: {list(unified.columns)}")
    
    # Report source distribution
    source_counts = unified['source'].value_counts()
    print(f"[Unify] Source distribution:")
    for source, count in source_counts.items():
        print(f"        - {source}: {count:,}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    parquet_path = f'{args.output}.parquet'
    unified.to_parquet(parquet_path, index=False)
    print(f"[Unify] Saved: {parquet_path}")
    
    if args.export_csv:
        csv_path = f'{args.output}.csv'
        unified.to_csv(csv_path, index=False, lineterminator='\n', encoding='utf-8')
        print(f"[Unify] Saved: {csv_path}")


if __name__ == '__main__':
    main()
