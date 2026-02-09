#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engineer Features: Physicochemical Property Calculation
========================================================

Calculates physical and chemical features for aqueous solubility prediction.
Implements domain-specific feature engineering including temperature-aware
sample weighting, ionization state counting, and pKa prediction.

Features Computed:
    - weight: Balanced sample weights (temperature Gaussian + solubility rarity)
    - n_ionizable: Total count of ionizable groups
    - n_acid: Number of acidic ionization sites (via SMARTS)
    - n_base: Number of basic ionization sites (via SMARTS)
    - n_phenol: Count of phenolic hydroxyl groups
    - has_phenol: Binary indicator for phenol presence
    - pka_acid_min/max: Predicted acidic pKa range (via MolGpKa API)
    - pka_base_min/max: Predicted basic pKa range (via MolGpKa API)

Arguments:
    --in              : Input Parquet/CSV file with molecular data
    --out             : Output Parquet file path
    --smiles-col      : Column name containing SMILES strings (default: smiles_neutral)
    --temp-col        : Column name containing temperature in Celsius (default: temp_C)
    --smarts          : Path to SMARTS patterns file for ionization detection
    --pka-api-url     : URL for pKa prediction API (default: MolGpKa endpoint)
    --pka-token       : Authentication token for pKa API
    --nproc           : Number of CPUs for parallel processing (default: 1)

Usage:
    python engineer_features.py \\
        --in standardized.parquet \\
        --out engineered_features.parquet \\
        --smiles-col smiles_neutral \\
        --temp-col temp_C \\
        --smarts resources/smarts_pattern_ionized.txt \\
        --pka-api-url "http://xundrug.cn:5001/modules/upload0/" \\
        --pka-token "YOUR_TOKEN" \\
        --nproc 4

Output:
    Parquet file with original columns plus engineered features:
    - weight, n_ionizable, n_acid, n_base, n_phenol, has_phenol
    - pka_acid_min, pka_acid_max, pka_base_min, pka_base_max (if API enabled)

Notes:
    - Sample weighting uses Gaussian centered at 37°C (physiological temperature)
    - pKa values are imputed for neutral molecules (acid=16.0, base=-2.0)
    - Parallel processing available for ionization and phenol calculations
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import requests

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem


# ============================================================================
# PARALLEL PROCESSING WRAPPERS
# ============================================================================

def _process_chunk_ionization(chunk_tuple):
    """
    Process a chunk of SMILES for ionization calculation in parallel.
    
    Args:
        chunk_tuple: Tuple of (chunk_index, smiles_series, acid_df, base_df)
    
    Returns:
        Tuple of (chunk_index, results_series)
    """
    idx, series, df_acid, df_base = chunk_tuple
    res = series.apply(lambda x: calculate_ionization(x, df_acid, df_base))
    return idx, res


def _process_chunk_phenols(chunk_tuple):
    """
    Process a chunk of SMILES for phenol calculation in parallel.
    
    Args:
        chunk_tuple: Tuple of (chunk_index, smiles_series)
    
    Returns:
        Tuple of (chunk_index, results_series)
    """
    idx, series = chunk_tuple
    res = series.apply(calculate_phenols)
    return idx, res


# ============================================================================
# 1. SAMPLE WEIGHTS (Temperature + Solubility Rarity)
# ============================================================================

def calculate_balanced_weights(
    df: pd.DataFrame,
    temp_col: str = 'temp_C',
    target_col: str = 'logS',
    target_temp: float = 37.0,
    sigma: float = 5.0
) -> pd.Series:
    """
    Calculate sample weights prioritizing physiological temperature (37°C).
    
    Weighting Strategy:
    1. Temperature Weight: Gaussian centered at target_temp with given sigma
       - At 37°C: weight = 5.0 (base + bonus)
       - At 25°C: weight ≈ 1.2 (mostly base)
    
    2. Solubility Rarity Weight: Log-scaled inverse frequency
       - Rare logS bins get higher weights
       - Prevents model bias toward common solubility ranges
    
    3. Final Weight: w_solubility × (w_temp)²
       - Quadratic scaling aggressively prioritizes 37°C data
       - A sample at 37°C is ~17× more important than one at 25°C
    
    Args:
        df: DataFrame with temperature and target columns
        temp_col: Column name for temperature in Celsius
        target_col: Column name for solubility (logS)
        target_temp: Target temperature for Gaussian peak (default: 37.0)
        sigma: Standard deviation for Gaussian (default: 5.0)
    
    Returns:
        Series of sample weights (same index as input df)
    """
    print(f"[Features] Calculating Balanced Weights (Target Temp={target_temp}°C + LogS Rarity)...")
    
    # --- Temperature Weight (Gaussian) ---
    # Fill missing temperatures with 25°C (room temperature assumption)
    t = pd.to_numeric(df[temp_col], errors='coerce').fillna(25.0)
    
    base_weight = 1.0
    bonus_weight = 4.0  # Peak bonus at target temperature
    
    # Gaussian calculation
    # At 37°C: exp(0) = 1.0 → w = 1.0 + 4.0 = 5.0
    # At 25°C: exp(-2.88) ≈ 0.05 → w = 1.0 + 0.2 = 1.2
    exp_term = np.exp(-((t - target_temp) ** 2) / (2 * sigma ** 2))
    w_temp = base_weight + (bonus_weight * exp_term)
    
    # --- Solubility Rarity Weight ---
    if target_col in df.columns:
        logs = pd.to_numeric(df[target_col], errors='coerce').fillna(-2.0)
        
        # Create 16 bins from -12 to +4 logS
        bins = np.linspace(-12, 4, 17)
        counts, _ = np.histogram(logs, bins=bins)
        counts = np.maximum(counts, 1)  # Avoid division by zero
        
        total_samples = len(df)
        
        # Map each sample to its bin
        indices = np.digitize(logs, bins) - 1
        indices = np.clip(indices, 0, len(counts) - 1)
        row_counts = counts[indices]
        
        # Log-scaled inverse frequency: log(1 + total/count)
        # Typical range: 1.0 to 5.0
        w_solubility = np.log1p(total_samples / row_counts)
        
        print(f"   -> Solubility Weights range: {w_solubility.min():.2f} - {w_solubility.max():.2f}")
    else:
        print(f"[WARN] Target column '{target_col}' not found. Using only Temperature weights.")
        w_solubility = 1.0
    
    # --- Combine with Quadratic Scaling ---
    # Quadratic temperature scaling severely penalizes 25°C bias
    # Ratio: 37°C → 5² = 25, 25°C → 1.2² = 1.44 → 17× difference
    final_weight = w_solubility * (w_temp ** 2)
    
    print(f"   -> Final Weights (Mean): {final_weight.mean():.2f}")
    return final_weight.fillna(1.0)


# ============================================================================
# 2. IONIZATION DETECTION (SMARTS Patterns)
# ============================================================================

def split_acid_base_pattern(smarts_file: str) -> tuple:
    """
    Load and split SMARTS patterns into acid and base DataFrames.
    
    Args:
        smarts_file: Path to TSV file with SMARTS patterns
                     Expected columns: SMARTS, Acid_or_base, new_index
    
    Returns:
        Tuple of (acid_df, base_df)
    """
    try:
        df_smarts = pd.read_csv(smarts_file, sep="\t")
    except FileNotFoundError:
        sys.exit(f"[ERROR] SMARTS file not found: {smarts_file}")
    
    df_acid = df_smarts[df_smarts.Acid_or_base == "A"].reset_index(drop=True)
    df_base = df_smarts[df_smarts.Acid_or_base == "B"].reset_index(drop=True)
    
    return df_acid, df_base


def unique_acid_match(matches: list) -> list:
    """
    Deduplicate atom matches from multiple SMARTS patterns.
    
    Args:
        matches: List of atom index tuples from substructure matching
    
    Returns:
        Deduplicated list of matches
    """
    single_matches = list(set([m[0] for m in matches if len(m) == 1]))
    double_matches = [m for m in matches if len(m) == 2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches


def match_pattern(df_smarts: pd.DataFrame, mol) -> list:
    """
    Find all ionizable atom indices matching SMARTS patterns.
    
    Args:
        df_smarts: DataFrame with SMARTS patterns and atom indices
        mol: RDKit Mol object (with explicit hydrogens)
    
    Returns:
        List of unique atom indices that match ionizable patterns
    """
    matches = []
    
    for row in df_smarts.itertuples():
        try:
            smarts_str = row.SMARTS
            index_str = row.new_index
        except AttributeError:
            continue
        
        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None:
            continue
        
        match = mol.GetSubstructMatches(pattern)
        if not match:
            continue
        
        # Handle multi-atom patterns (e.g., "0,1" for carboxylic acid)
        if isinstance(index_str, str) and "," in index_str:
            idxs = [int(i) for i in index_str.split(",")]
            for m in match:
                matches.append([m[idxs[0]], m[idxs[1]]])
        else:
            try:
                idx = int(index_str)
            except (ValueError, TypeError):
                continue
            for m in match:
                if idx < len(m):
                    matches.append([m[idx]])
    
    matches = unique_acid_match(matches)
    flat_matches = [item for sublist in matches for item in sublist]
    return list(set(flat_matches))


def calculate_ionization(smiles: str, df_acid: pd.DataFrame, df_base: pd.DataFrame) -> pd.Series:
    """
    Count ionizable groups (acid and base) in a molecule.
    
    Args:
        smiles: SMILES string of the molecule
        df_acid: DataFrame with acidic SMARTS patterns
        df_base: DataFrame with basic SMARTS patterns
    
    Returns:
        Series with n_ionizable, n_acid, n_base counts
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series({"n_ionizable": 0, "n_acid": 0, "n_base": 0})
    
    # Add explicit hydrogens for accurate matching
    mol = AllChem.AddHs(mol)
    
    acid_idx = match_pattern(df_acid, mol)
    base_idx = match_pattern(df_base, mol)
    
    return pd.Series({
        "n_ionizable": int(len(acid_idx) + len(base_idx)),
        "n_acid": int(len(acid_idx)),
        "n_base": int(len(base_idx))
    })


# ============================================================================
# 3. PHENOL DETECTION
# ============================================================================

# SMARTS pattern for phenolic hydroxyl: aromatic carbon bonded to OH
_PHENOL_SMARTS = "c[OX2H]"


def calculate_phenols(smiles: str) -> pd.Series:
    """
    Count phenolic hydroxyl groups in a molecule.
    
    Args:
        smiles: SMILES string of the molecule
    
    Returns:
        Series with n_phenol count and has_phenol binary flag
    """
    phenol_pattern = Chem.MolFromSmarts(_PHENOL_SMARTS)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series({"n_phenol": 0, "has_phenol": 0})
    
    matches = mol.GetSubstructMatches(phenol_pattern)
    
    # Count unique oxygen atoms (index 1 in the SMARTS match)
    n = len({m[1] for m in matches if len(m) > 1}) if matches else 0
    
    return pd.Series({
        "n_phenol": int(n),
        "has_phenol": int(n > 0)
    })


# ============================================================================
# 4. pKa PREDICTION (MolGpKa API)
# ============================================================================

def predict_pka_api(smiles: str, api_url: str, token: str) -> dict:
    """
    Query external MolGpKa API for pKa prediction.
    
    Args:
        smiles: SMILES string of the molecule
        api_url: API endpoint URL
        token: Authentication token
    
    Returns:
        Dictionary with 'Acid' and 'Base' pKa predictions, or None on failure
    """
    if not smiles:
        return None
    
    try:
        resp = requests.post(
            url=api_url,
            files={"Smiles": ("tmg", smiles)},
            headers={"token": token},
            timeout=20
        )
        
        if resp.status_code != 200:
            return None
        
        res_json = resp.json()
        if res_json.get("status") != 200:
            return None
        
        return res_json.get("gen_datas")
    
    except (requests.RequestException, ValueError):
        return None


def summarize_pka(val: dict) -> pd.Series:
    """
    Extract min/max pKa values from API response.
    
    Args:
        val: Dictionary with 'Acid' and 'Base' sub-dictionaries
    
    Returns:
        Series with pka_acid_min, pka_acid_max, pka_base_min, pka_base_max
    """
    default = {
        "pka_acid_min": np.nan,
        "pka_acid_max": np.nan,
        "pka_base_min": np.nan,
        "pka_base_max": np.nan
    }
    
    if not val or not isinstance(val, dict):
        return pd.Series(default)
    
    acid = val.get("Acid", {})
    base = val.get("Base", {})
    
    def get_vals(d):
        return [float(v) for v in d.values() if v]
    
    a_vals = get_vals(acid)
    b_vals = get_vals(base)
    
    return pd.Series({
        "pka_acid_min": min(a_vals) if a_vals else np.nan,
        "pka_acid_max": max(a_vals) if a_vals else np.nan,
        "pka_base_min": min(b_vals) if b_vals else np.nan,
        "pka_base_max": max(b_vals) if b_vals else np.nan
    })


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for feature engineering script."""
    
    ap = argparse.ArgumentParser(
        description="Calculate physicochemical features for solubility prediction."
    )
    ap.add_argument("--in", dest="inp", required=True,
                    help="Input Parquet/CSV file")
    ap.add_argument("--out", dest="out", required=True,
                    help="Output Parquet file")
    ap.add_argument("--smiles-col", default="smiles_neutral",
                    help="SMILES column name (default: smiles_neutral)")
    ap.add_argument("--temp-col", default="temp_C",
                    help="Temperature column name (default: temp_C)")
    ap.add_argument("--smarts", required=True,
                    help="Path to SMARTS patterns file for ionization")
    ap.add_argument("--pka-api-url", default="http://xundrug.cn:5001/modules/upload0/",
                    help="pKa prediction API URL")
    ap.add_argument("--pka-token", default=None,
                    help="pKa API authentication token")
    ap.add_argument("--nproc", type=int, default=1,
                    help="Number of CPUs for parallel processing")
    
    args = ap.parse_args()
    
    # Configure parallelization
    n_jobs = max(1, args.nproc)
    print(f"[Features] Running with {n_jobs} CPUs")
    
    # -------------------------------------------------------------------------
    # Load Input Data
    # -------------------------------------------------------------------------
    print(f"[Features] Loading {args.inp}...")
    if args.inp.endswith(".parquet"):
        df = pd.read_parquet(args.inp)
    else:
        df = pd.read_csv(args.inp)
    
    if args.smiles_col not in df.columns:
        sys.exit(f"[ERROR] SMILES column '{args.smiles_col}' not found in input file.")
    
    print(f"[Features] Loaded {len(df):,} rows")
    
    # -------------------------------------------------------------------------
    # 1. Calculate Sample Weights
    # -------------------------------------------------------------------------
    df["weight"] = calculate_balanced_weights(df, args.temp_col, target_col='logS')
    
    # -------------------------------------------------------------------------
    # 2. Calculate Ionization Features
    # -------------------------------------------------------------------------
    df_acid, df_base = split_acid_base_pattern(args.smarts)
    
    print("[Features] Calculating Ionization...")
    if n_jobs > 1:
        # Parallel processing with chunks
        chunks = np.array_split(df[args.smiles_col], n_jobs * 4)
        
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(_process_chunk_ionization, (i, chunk, df_acid, df_base))
                for i, chunk in enumerate(chunks)
            ]
            
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Ionization"):
                results.append(future.result())
        
        # Reorder and concatenate
        results.sort(key=lambda x: x[0])
        ion_feats = pd.concat([res[1] for res in results])
    else:
        tqdm.pandas(desc="Ionization")
        ion_feats = df[args.smiles_col].progress_apply(
            lambda x: calculate_ionization(x, df_acid, df_base)
        )
    
    df = pd.concat([df, ion_feats], axis=1)
    
    # -------------------------------------------------------------------------
    # 3. Calculate Phenol Features
    # -------------------------------------------------------------------------
    print("[Features] Calculating Phenols...")
    if n_jobs > 1:
        chunks = np.array_split(df[args.smiles_col], n_jobs * 4)
        
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(_process_chunk_phenols, (i, chunk))
                for i, chunk in enumerate(chunks)
            ]
            
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Phenols"):
                results.append(future.result())
        
        results.sort(key=lambda x: x[0])
        phenol_feats = pd.concat([res[1] for res in results])
    else:
        tqdm.pandas(desc="Phenols")
        phenol_feats = df[args.smiles_col].progress_apply(calculate_phenols)
    
    df = pd.concat([df, phenol_feats], axis=1)
    
    # -------------------------------------------------------------------------
    # 4. Query pKa API (Optional)
    # -------------------------------------------------------------------------
    if args.pka_token:
        print("[Features] Querying pKa API...")
        unique_smiles = df[args.smiles_col].unique()
        cache = {}
        
        # Use ThreadPoolExecutor for I/O-bound API calls
        # Limit threads to avoid overwhelming the API server
        api_threads = min(8, n_jobs * 2)
        
        with ThreadPoolExecutor(max_workers=api_threads) as executor:
            future_to_smi = {
                executor.submit(predict_pka_api, str(smi), args.pka_api_url, args.pka_token): str(smi)
                for smi in unique_smiles
            }
            
            for future in tqdm(as_completed(future_to_smi), total=len(unique_smiles), desc="pKa API"):
                smi = future_to_smi[future]
                try:
                    res = future.result()
                    cache[smi] = summarize_pka(res)
                except Exception:
                    cache[smi] = summarize_pka(None)
        
        # Map results back to DataFrame
        pka_results = df[args.smiles_col].astype(str).map(cache).apply(pd.Series)
        
        # Impute missing values (neutral molecules)
        # Acid pKa = 16.0 (weaker than water)
        # Base pKa = -2.0 (weaker than hydronium)
        pka_results["pka_acid_min"] = pka_results["pka_acid_min"].fillna(16.0)
        pka_results["pka_acid_max"] = pka_results["pka_acid_max"].fillna(16.0)
        pka_results["pka_base_min"] = pka_results["pka_base_min"].fillna(-2.0)
        pka_results["pka_base_max"] = pka_results["pka_base_max"].fillna(-2.0)
        
        df = pd.concat([df, pka_results], axis=1)
    else:
        print("[INFO] Skipping pKa calculation (no API token provided).")
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    print(f"[Features] Saving to {args.out}...")
    df.to_parquet(args.out, index=False)
    
    print(f"[Features] Done. Output shape: {df.shape}")


if __name__ == "__main__":
    main()
