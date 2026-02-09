#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
    fetch_chembl.py
========================================================================================
    Processes ChEMBL solubility data for pipeline integration.
    
    ChEMBL is a manually curated database of bioactive molecules with drug-like
    properties. This script extracts solubility measurements from a local ChEMBL
    export and converts them to standardized logS values.
    
    Processing Pipeline:
    1. Load ChEMBL CSV export (auto-detects delimiter)
    2. Convert diverse solubility units to logS (mol/L)
    3. Filter for poorly soluble compounds (logS < -5)
    4. Extract temperature from assay descriptions using regex
    5. Format output for downstream processing
    
    Supported Units:
    - Molar concentrations: M, mM, µM, nM
    - Mass/volume: mg/mL, µg/mL, ng/mL, g/L
    - Pre-computed logarithmic values

    Arguments:
        --input     Input ChEMBL CSV file (default: chembl_raw.csv)
        --output    Output CSV file (default: chembl.csv)
        --threshold LogS threshold for filtering (default: -5.0)
    
    Usage:
        python fetch_chembl.py --input chembl_raw.csv
    
    Output:
        chembl.csv with columns: smiles_original, logS, solvent, temp_C, source, is_temp_assumed
----------------------------------------------------------------------------------------
"""

import argparse
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ======================================================================================
#     CONSTANTS
# ======================================================================================

# Default temperature when not specified in assay (room temperature)
DEFAULT_TEMPERATURE_C = 25.0

# LogS threshold for "poorly soluble" compounds
# Compounds with logS < -5 are considered practically insoluble
INSOLUBILITY_THRESHOLD = -5.0

# ======================================================================================
#     TEMPERATURE EXTRACTION
# ======================================================================================

def extract_temp_from_text(text: str) -> float:
    """
    Extract temperature from assay description text.
    
    Uses regex patterns to identify temperature mentions in various formats:
    - "37 deg C", "25 degrees", "37C"
    - "room temperature", "ambient"
    
    Args:
        text: Assay description text
        
    Returns:
        Temperature in Celsius (defaults to 25°C if not found)
    """
    if pd.isna(text) or text == '':
        return DEFAULT_TEMPERATURE_C
    
    text = str(text)
    
    # Pattern: digits followed by degree indicators
    match = re.search(r'(\d{2,3})\s?(?:deg|degrees|°|C)\b', text, re.IGNORECASE)
    if match:
        temp = float(match.group(1))
        # Sanity check: temperature should be reasonable (0-100°C)
        if 0 <= temp <= 100:
            return temp
    
    # Check for room temperature keywords
    lower_text = text.lower()
    if "room temp" in lower_text or "ambient" in lower_text:
        return DEFAULT_TEMPERATURE_C
    
    return DEFAULT_TEMPERATURE_C

# ======================================================================================
#     UNIT CONVERSION
# ======================================================================================

def convert_to_logS_molar(row: pd.Series) -> Optional[float]:
    """
    Convert solubility value and unit to logS (log10 of molar concentration).
    
    Handles multiple unit types:
    1. Molar units (M, mM, µM, nM) - direct conversion
    2. Mass/volume units (mg/mL, etc.) - requires molecular weight
    3. Pre-computed logS values - passed through
    
    Args:
        row: DataFrame row with 'Standard Value', 'Standard Units', 
             'Molecular Weight', and optionally 'Standard Type'
             
    Returns:
        logS value (log10 mol/L) or None if conversion fails
    """
    # Extract and validate value
    try:
        value = float(row['Standard Value'])
        if value <= 0:
            return None
    except (ValueError, TypeError):
        return None

    unit = str(row['Standard Units']).lower().strip()
    
    # -------------------------------------------------------------------------
    # Case 1: Molar concentration units (no MW needed)
    # -------------------------------------------------------------------------
    molar_conversions = {
        'm': 1.0,           # Molar
        'molar': 1.0,
        'mm': 1e-3,         # Millimolar
        'um': 1e-6,         # Micromolar
        'µm': 1e-6,
        'nm': 1e-9,         # Nanomolar
    }
    
    if unit in molar_conversions:
        molar_conc = value * molar_conversions[unit]
        return np.log10(molar_conc) if molar_conc > 0 else None
    
    # -------------------------------------------------------------------------
    # Case 2: Mass/volume units (requires molecular weight)
    # -------------------------------------------------------------------------
    try:
        mw = float(row['Molecular Weight'])
        if mw <= 0:
            return None
    except (ValueError, TypeError):
        return None
    
    # Convert to g/L first, then to molar
    mass_volume_conversions = {
        'ug.ml-1': 1e-3,    # µg/mL -> g/L
        'ug/ml': 1e-3,
        'mcg/ml': 1e-3,
        'ng/ml': 1e-6,      # ng/mL -> g/L
        'mg/ml': 1.0,       # mg/mL = g/L
        'mg.l-1': 1e-3,     # mg/L -> g/L
        'g/l': 1.0,         # Already g/L
    }
    
    if unit in mass_volume_conversions:
        conc_g_L = value * mass_volume_conversions[unit]
        molar_conc = conc_g_L / mw  # g/L ÷ g/mol = mol/L
        return np.log10(molar_conc) if molar_conc > 0 else None
    
    # -------------------------------------------------------------------------
    # Case 3: Pre-computed logS (unit missing but type indicates log)
    # -------------------------------------------------------------------------
    std_type = str(row.get('Standard Type', '')).lower()
    if 'log' in std_type and unit in ('nan', '', 'none'):
        return value  # Already in log scale
    
    return None

# ======================================================================================
#     MAIN PROCESSING
# ======================================================================================

def main() -> None:
    """Main entry point for ChEMBL data processing."""
    parser = argparse.ArgumentParser(
        description="Process ChEMBL solubility data for pipeline integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", 
        default="chembl_raw.csv",
        help="Input ChEMBL CSV file (default: chembl_raw.csv)"
    )
    parser.add_argument(
        "--output",
        default="chembl.csv",
        help="Output CSV file (default: chembl.csv)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=INSOLUBILITY_THRESHOLD,
        help=f"LogS threshold for filtering (default: {INSOLUBILITY_THRESHOLD})"
    )
    args = parser.parse_args()

    print(f"[ChEMBL] Processing: {args.input}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load CSV
    # -------------------------------------------------------------------------
    try:
        # ChEMBL exports may use different delimiters; auto-detect
        df = pd.read_csv(args.input, sep=None, engine='python')
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        sys.exit(1)

    print(f"[ChEMBL] Original rows: {len(df):,}")
    
    # -------------------------------------------------------------------------
    # Step 2: Prepare data
    # -------------------------------------------------------------------------
    # Ensure numeric columns
    df['Molecule Max Phase'] = pd.to_numeric(
        df['Molecule Max Phase'], 
        errors='coerce'
    ).fillna(0)
    
    df_compounds = df.copy()
    print(f"[ChEMBL] Total compounds: {len(df_compounds):,}")

    # -------------------------------------------------------------------------
    # Step 3: Convert to logS
    # -------------------------------------------------------------------------
    # Use tqdm for progress if available
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Converting units")
        df_compounds['logS'] = df_compounds.progress_apply(convert_to_logS_molar, axis=1)
    except ImportError:
        df_compounds['logS'] = df_compounds.apply(convert_to_logS_molar, axis=1)

    # Remove failed conversions
    df_valid = df_compounds.dropna(subset=['logS']).copy()
    print(f"[ChEMBL] Valid logS values: {len(df_valid):,}")

    # -------------------------------------------------------------------------
    # Step 4: Filter poorly soluble compounds
    # -------------------------------------------------------------------------
    df_insoluble = df_valid[df_valid['logS'] < args.threshold].copy()
    print(f"[ChEMBL] Poorly soluble (logS < {args.threshold}): {len(df_insoluble):,}")

    # -------------------------------------------------------------------------
    # Step 5: Extract temperature
    # -------------------------------------------------------------------------
    df_insoluble['temp_C'] = df_insoluble['Assay Description'].apply(extract_temp_from_text)
    
    # Flag assumed temperatures (where default was used)
    # Note: This is a simplification - ideally we'd track if extract returned default
    df_insoluble['is_temp_assumed'] = (df_insoluble['temp_C'] == DEFAULT_TEMPERATURE_C).astype(int)

    # -------------------------------------------------------------------------
    # Step 6: Format output
    # -------------------------------------------------------------------------
    output_df = pd.DataFrame({
        'smiles_original': df_insoluble['Smiles'],
        'logS': df_insoluble['logS'],
        'solvent': 'water',
        'temp_C': df_insoluble['temp_C'],
        'source': 'ChEMBL_Phase_' + df_insoluble['Molecule Max Phase'].astype(int).astype(str),
        'is_temp_assumed': df_insoluble['is_temp_assumed']
    })
    
    # Remove entries without SMILES
    output_df = output_df.dropna(subset=['smiles_original'])
    
    # Save output
    output_df.to_csv(args.output, index=False)
    print(f"[ChEMBL] Saved: {args.output} ({len(output_df):,} compounds)")


if __name__ == "__main__":
    main()
