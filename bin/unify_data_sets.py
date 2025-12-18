#!/usr/bin/env python3
import argparse
import sys
import os
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
    'is_outlier',
    'source'
]

# Column name mapping candidates
# NOTE: We split temperature into C and K to handle conversions automatically
CANDIDATES = {
    'smiles_original': ['smiles_original', 'canonical_smiles', 'SMILES', 'smiles', 'SMILES_Solute'],
    'smiles_neutral' : ['smiles_neutral'],
    'solvent'        : ['solvent', 'Solvent', 'Medium', 'SMILES_Solvent', 'Solvent (Solubility (MCS))'],
    
    # Nombres posibles para columnas que YA están en Celsius
    'temp_C_sources' : ['temp_C', 'Temperature_C', 'T_C', 'Temp_C'],
    
    # Nombres posibles para columnas en Kelvin (que necesitan resta)
    'temp_K_sources' : ['temperature_K', 'Temperature_K', 'temp_K', 'T_K', 'T'],

    'logS'           : ['logS', 'LogS', 'logS_molL', 'Solubility', 'Y', 'LogS(mol/L)'],
    'is_outlier'     : ['is_outlier', 'outlier'],
    'source': ['source', 'Source', 'dataset', 'origin'],
}

def read_table(path: str) -> pd.DataFrame:
    """Reads CSV or Parquet with fallbacks for bad CSV encoding."""
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except Exception:
        # Fallback for problematic engines or separators (try tab if comma fails implicitly handled by python engine sniffing sometimes, but explicit is better)
        try:
             return pd.read_csv(path, sep='\t', engine='python', encoding_errors='ignore')
        except:
             return pd.read_csv(path, engine='python', encoding_errors='ignore')

def pick_series(df: pd.DataFrame, names: List[str]) -> pd.Series:
    """Searches for the first available column name from the list."""
    for c in names:
        if c in df.columns:
            return df[c]
    # Return empty series if not found (all NaNs)
    return pd.Series([np.nan] * len(df), index=df.index)

def coerce_bool_nullable(s: pd.Series) -> pd.Series:
    """Robust boolean conversion (handles 'yes', '1', 'true', etc.)."""
    # If already boolean, return
    if pd.api.types.is_bool_dtype(s):
        return s
    
    def _conv(x):
        if pd.isna(x): return np.nan
        t = str(x).strip().lower()
        if t in ('true', '1', 'yes', 'y', 't'):  return True
        if t in ('false', '0', 'no', 'n', 'f'):  return False
        return np.nan
        
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
    current_source = pick_series(raw, CANDIDATES.get('source', []))

    if current_source.isna().all():
        # Si no hay columna source, usamos el nombre del archivo (ej: 'reaxys.csv')
        filename = os.path.basename(path)
        out['source'] = filename
    else:
        out['source'] = current_source.astype(str)

    # 2. Numeric Columns (Smart Temperature Logic)
    # Intenta buscar Celsius directo
    raw_C = to_numeric_nullable(pick_series(raw, CANDIDATES['temp_C_sources']))
    
    # Intenta buscar Kelvin
    raw_K = to_numeric_nullable(pick_series(raw, CANDIDATES['temp_K_sources']))
    converted_C = raw_K - 273.15
    
    # Lógica de fusión:
    # Si tenemos dato en Celsius, úsalo. Si es NaN, mira si tenemos dato en Kelvin convertido.
    # fillna() hace exactamente eso.
    out['temp_C'] = raw_C.fillna(converted_C).round(2)

    # 3. Target & Flags
    out['logS']       = to_numeric_nullable(pick_series(raw, CANDIDATES['logS']))
    out['is_outlier'] = coerce_bool_nullable(pick_series(raw, CANDIDATES['is_outlier']))

    # 4. Ensure all NEEDED columns exist (fill NA if missing)
    for col in NEEDED:
        if col not in out.columns:
            out[col] = pd.Series([np.nan]*len(out), index=out.index)
    
    # Filter to keep only the strictly needed schema
    out = out[NEEDED]

    # 5. Final Type Enforcement
    for col in ['smiles_original', 'smiles_neutral', 'solvent']:
        out[col] = out[col].astype('string')
    for col in ['temp_C', 'logS']:
        out[col] = to_numeric_nullable(out[col])

    out['source'] = out['source'].astype('string')
    
    # Drop rows where absolutely crucial data (logS) is missing? 
    # Usually better to keep specific logic outside, but here we just return the frame.

    return out

def main():
    ap = argparse.ArgumentParser(description="Unify multiple solubility datasets into a standard schema.")
    ap.add_argument('--sources', nargs='+', required=True, help='Input CSV/Parquet files (accepts wildcards like *.csv)')
    ap.add_argument('--export-csv', action='store_true', help='Export unified.csv in addition to Parquet')
    args = ap.parse_args()

    parts = []
    for f in args.sources:
        try:
            print(f"[Unify] Processing {f} ...")
            dfp = load_one(f)
            
            # Simple check to see what temperature happened
            valid_temps = dfp['temp_C'].notna().sum()
            print(f"   -> Rows: {len(dfp)} | Valid Temps: {valid_temps}")
            
            parts.append(dfp)
        except Exception as e:
            print(f"[Unify] WARN: Failed to process {f}: {e}", file=sys.stderr)

    # Handle empty case
    if not parts:
        print("[Unify] No valid data found. Creating empty dataset.")
        # Create empty dummy with correct types
        dummy = pd.DataFrame(columns=NEEDED)
        dummy.to_parquet('unified.parquet', index=False)
        if args.export_csv:
            dummy.to_csv('unified.csv', index=False)
        sys.exit(0)

    # Concatenate all sources
    df = pd.concat(parts, axis=0, ignore_index=True)
    print("-" * 40)
    print(f"[Unify] Total rows after unification: {len(df)}")
    print(f"[Unify] Columns: {list(df.columns)}")

    # Save
    df.to_parquet('unified.parquet', index=False)
    print(f"[Unify] Saved to unified.parquet")
    
    if args.export_csv:
        df.to_csv('unified.csv', index=False, lineterminator='\n', encoding='utf-8')
        print(f"[Unify] Saved to unified.csv")

if __name__ == '__main__':
    main()