#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
import numpy as np
from typing import List

# --- CONFIGURATION ---

# Columns that MUST exist in the final output
NEEDED = [
    'smiles_original',
    'smiles_neutral',
    'solvent',
    'temp_C',
    'logS',
    'is_outlier'
]

# Column name mapping candidates (Order denotes priority)
CANDIDATES = {
    'smiles_original': ['smiles_original', 'canonical_smiles', 'SMILES', 'smiles', 'SMILES_Solute'],
    'smiles_neutral' : ['smiles_neutral'],
    'solvent'        : ['solvent', 'Solvent', 'Medium', 'SMILES_Solvent'],
    'temperature_K'  : ['temperature_K', 'Temperature_K', 'temperature', 'Temperature', 'temp_K', 'T_K', 'T'],
    'logS'           : ['logS', 'LogS', 'logS_molL', 'Solubility', 'Y', 'LogS(mol/L)'],
    'is_outlier'     : ['is_outlier', 'outlier'],
}

def read_table(path: str) -> pd.DataFrame:
    """Reads CSV or Parquet with fallbacks for bad CSV encoding."""
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except Exception:
        # Fallback for problematic engines
        return pd.read_csv(path, engine='python', encoding_errors='ignore')

def pick_series(df: pd.DataFrame, names: List[str]) -> pd.Series:
    """Searches for the first available column name from the list."""
    for c in names:
        if c in df.columns:
            return df[c]
    # Return empty series if not found
    return pd.Series([pd.NA] * len(df), index=df.index)

def coerce_bool_nullable(s: pd.Series) -> pd.Series:
    """Robust boolean conversion (handles 'yes', '1', 'true', etc.)."""
    if s.dtype == 'boolean':
        return s
    
    def _conv(x):
        if pd.isna(x): return pd.NA
        t = str(x).strip().lower()
        if t in ('true', '1', 'yes', 'y', 't'):  return True
        if t in ('false', '0', 'no', 'n', 'f'):  return False
        return pd.NA
        
    return s.apply(_conv).astype('boolean')

def to_numeric_nullable(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')

def load_one(path: str) -> pd.DataFrame:
    """Loads and standardizes a single dataset file."""
    raw = read_table(path)
    out = pd.DataFrame(index=raw.index)

    # 1. String Columns
    out['smiles_original'] = pick_series(raw, CANDIDATES['smiles_original']).astype('string')
    out['smiles_neutral']  = pick_series(raw, CANDIDATES['smiles_neutral']).astype('string')
    out['solvent']         = pick_series(raw, CANDIDATES['solvent']).astype('string')

    # 2. Numeric Columns (Temperature conversions)
    tempK = to_numeric_nullable(pick_series(raw, CANDIDATES['temperature_K']))
    # Kelvin to Celsius
    out['temp_C'] = (tempK - 273.15).round(2)

    # 3. Target & Flags
    out['logS']       = to_numeric_nullable(pick_series(raw, CANDIDATES['logS']))
    out['is_outlier'] = coerce_bool_nullable(pick_series(raw, CANDIDATES['is_outlier']))

    # 4. Ensure all NEEDED columns exist (fill NA if missing)
    for col in NEEDED:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA]*len(out), index=out.index)
    
    # Filter to keep only the strictly needed schema
    out = out[NEEDED]

    # 5. Final Type Enforcement
    for col in ['smiles_original', 'smiles_neutral', 'solvent']:
        out[col] = out[col].astype('string')
    for col in ['temp_C', 'logS']:
        out[col] = to_numeric_nullable(out[col])

    return out

def main():
    ap = argparse.ArgumentParser(description="Unify multiple solubility datasets into a standard schema.")
    ap.add_argument('--sources', nargs='+', required=True, help='Input CSV/Parquet files')
    ap.add_argument('--export-csv', action='store_true', help='Export unified.csv in addition to Parquet')
    args = ap.parse_args()

    parts = []
    for f in args.sources:
        try:
            print(f"[Unify] Processing {f} ...")
            dfp = load_one(f)
            parts.append(dfp)
        except Exception as e:
            print(f"[Unify] WARN: Failed to process {f}: {e}", file=sys.stderr)

    # Handle empty case
    if not parts:
        print("[Unify] No valid data found. Creating empty dataset.")
        pd.DataFrame(columns=NEEDED).to_parquet('unified.parquet', index=False)
        if args.export_csv:
            pd.DataFrame(columns=NEEDED).to_csv('unified.csv', index=False)
        sys.exit(0)

    # Concatenate all sources
    df = pd.concat(parts, axis=0, ignore_index=True)
    print(f"[Unify] Total rows after unification: {len(df)}")

    # Save
    df.to_parquet('unified.parquet', index=False)
    if args.export_csv:
        df.to_csv('unified.csv', index=False, lineterminator='\n', encoding='utf-8')

if __name__ == '__main__':
    main()