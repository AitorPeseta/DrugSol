#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict Real Solubility: pH-Dependent Solubility Correction
============================================================

Calculates effective solubility at physiological pH (default 7.4) from the
model's intrinsic solubility prediction using thermodynamic corrections
based on ionization state.

Thermodynamic Correction:
    The model predicts Sw (experimental solubility at saturation pH).
    To get solubility at a different pH, we need to account for ionization:
    
    1. Calculate pH_sat (pH at saturation, where Sw was measured)
    2. Calculate S0 (intrinsic solubility of neutral form)
    3. Calculate fraction of neutral species at target pH
    4. Seff = S0 / f_neutral
    
    For zwitterions: The "neutral" form is the dipolar species (+/-)

Henderson-Hasselbalch Equations:
    Acid: fraction_neutral = 1 / (1 + 10^(pH - pKa))
    Base: fraction_neutral = 1 / (1 + 10^(pKa - pH))
    Zwitterion: fraction_neutral = 1 / (1 + 10^(pKa_acid - pH) + 10^(pH - pKa_base))

Arguments:
    --input  : CSV with 'smiles' and 'predicted_logS' columns
    --output : Output CSV filename
    --ph     : Target pH (default: 7.4)

Usage:
    python predict_real_solubility.py \\
        --input predictions.csv \\
        --output predictions_pH7.4.csv \\
        --ph 7.4

Output:
    CSV with additional columns: pka_acids, pka_bases, logSeff_pH{X}

Notes:
    - Requires external pKa API (xundrug.cn)
    - API failures return original prediction unchanged
    - Progress bar shown via tqdm
"""

import argparse
import time

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = 'http://xundrug.cn:5001/modules/upload0/'
API_TOKEN = 'O05DriqqQLlry9kmpCwms2IJLC0MuLQ7'
KW = 1.0e-14  # Water dissociation constant


# ============================================================================
# PKA FUNCTIONS
# ============================================================================

def get_pka_from_api(smiles: str, retries: int = 3) -> tuple:
    """
    Query pKa prediction API.
    
    Args:
        smiles: SMILES string
        retries: Number of retry attempts
    
    Returns:
        Tuple of (acid_pkas, base_pkas) lists
    """
    if not smiles or pd.isna(smiles):
        return [], []
    
    for _ in range(retries):
        try:
            response = requests.post(
                url=API_URL,
                files={"Smiles": ("tmg", smiles)},
                headers={'token': API_TOKEN},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json().get('gen_datas', {})
                acids = [float(v) for v in data.get('Acid', {}).values() if v]
                bases = [float(v) for v in data.get('Base', {}).values() if v]
                return acids, bases
        except Exception:
            time.sleep(1)
    
    return [], []


# ============================================================================
# THERMODYNAMIC FUNCTIONS
# ============================================================================

def net_charge(pH: float, acid_pkas: list, base_pkas: list, Sw: float) -> float:
    """
    Calculate net charge for finding saturation pH.
    
    Used with root-finding to determine the pH at which the molecule
    reaches its measured solubility (Sw).
    """
    h_conc = 10**(-pH)
    if h_conc == 0:
        return 0
    
    oh_conc = KW / h_conc
    charge = h_conc - oh_conc
    
    # Contribution from acidic groups
    for pka in acid_pkas:
        den = 1.0 + 10**(pka - pH)
        charge -= Sw * (1.0 / den)
    
    # Contribution from basic groups
    for pka in base_pkas:
        den = 1.0 + 10**(pH - pka)
        charge += Sw * (1.0 / den)
    
    return charge


def get_neutral_fraction(pH: float, acid_pkas: list, base_pkas: list) -> float:
    """
    Calculate fraction of neutral/zwitterionic species at given pH.
    
    Handles different molecule types:
    - Simple acids
    - Simple bases
    - Zwitterions (pKa_acid < pKa_base)
    - Regular ampholytes (pKa_base < pKa_acid)
    """
    if not acid_pkas and not base_pkas:
        return 1.0
    
    # Get most relevant pKa from each group
    pka_acid = min(acid_pkas) if acid_pkas else None
    pka_base = max(base_pkas) if base_pkas else None
    
    # Ampholytes (have both acid and base)
    if pka_acid is not None and pka_base is not None:
        # Zwitterion (acid pKa < base pKa): neutral form is dipolar (+/-)
        if pka_acid < pka_base:
            denom = 1.0 + 10**(pka_acid - pH) + 10**(pH - pka_base)
            return 1.0 / denom
        # Regular ampholyte: neutral form has no charge
        else:
            prob_acid_neutral = 1.0 / (1.0 + 10**(pH - pka_acid))
            prob_base_neutral = 1.0 / (1.0 + 10**(pka_base - pH))
            return prob_acid_neutral * prob_base_neutral
    
    # Simple acid
    if pka_acid is not None:
        return 1.0 / (1.0 + 10**(pH - pka_acid))
    
    # Simple base
    if pka_base is not None:
        return 1.0 / (1.0 + 10**(pka_base - pH))
    
    return 1.0


def process_molecule(row: pd.Series, target_ph: float) -> dict:
    """
    Calculate effective solubility at target pH for a single molecule.
    
    Args:
        row: DataFrame row with 'predicted_logS' and 'smiles'
        target_ph: Target pH for solubility calculation
    
    Returns:
        Dictionary with pKa values and calculated logSeff
    """
    try:
        log_sw = float(row['predicted_logS'])
        sw_molar = 10**log_sw
        smiles = row['smiles']
    except Exception:
        return None
    
    # Get pKa values
    acid_pkas, base_pkas = get_pka_from_api(smiles)
    
    # If no ionizable groups, solubility unchanged
    if not acid_pkas and not base_pkas:
        return {
            'pka_acids': "[]",
            'pka_bases': "[]",
            f'logSeff_pH{target_ph}': log_sw
        }
    
    # Calculate saturation pH (where Sw was theoretically measured)
    try:
        ph_sat = brentq(net_charge, -2, 16, args=(acid_pkas, base_pkas, sw_molar))
    except Exception:
        ph_sat = 7.0
    
    # Calculate intrinsic solubility S0 (mathematical bridge)
    f_neutral_sat = get_neutral_fraction(ph_sat, acid_pkas, base_pkas)
    s0_molar = sw_molar * f_neutral_sat
    
    # Numerical protection
    if s0_molar < 1e-15:
        s0_molar = 1e-15
    
    # Calculate effective solubility at target pH
    f_neutral_target = get_neutral_fraction(target_ph, acid_pkas, base_pkas)
    if f_neutral_target < 1e-12:
        f_neutral_target = 1e-12
    
    seff_molar = s0_molar / f_neutral_target
    log_seff = np.log10(seff_molar)
    
    return {
        'pka_acids': str(acid_pkas),
        'pka_bases': str(base_pkas),
        'pH_sat_calculated': round(ph_sat, 2),
        f'logSeff_pH{target_ph}': round(log_seff, 3)
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for physiological solubility calculation."""
    
    ap = argparse.ArgumentParser(
        description="Calculate effective solubility at target pH."
    )
    ap.add_argument("--input", required=True,
                    help="CSV with 'predicted_logS' column")
    ap.add_argument("--output", required=True,
                    help="Output CSV filename")
    ap.add_argument("--ph", type=float, default=7.4,
                    help="Target pH (default: 7.4)")
    
    args = ap.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"[Thermo] Calculating solubility at pH {args.ph} for {len(df):,} molecules...")
    
    # Process each molecule
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        res = process_molecule(row, args.ph)
        results.append(res if res else {})
    
    # Combine results
    res_df = pd.DataFrame(results)
    final_df = pd.concat([df, res_df], axis=1)
    
    # Reorder columns for readability
    cols = list(final_df.columns)
    important = ['smiles', 'predicted_logS', f'logSeff_pH{args.ph}', 'pka_acids', 'pka_bases']
    for c in reversed(important):
        if c in cols:
            cols.insert(0, cols.pop(cols.index(c)))
    
    final_df = final_df[cols]
    
    # Save
    final_df.to_csv(args.output, index=False)
    print(f"[Thermo] Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
