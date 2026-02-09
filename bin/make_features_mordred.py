#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Features Mordred: Comprehensive Molecular Descriptor Calculation
======================================================================

Calculates molecular descriptors using the Mordred library, providing a rich
feature set (~1800 descriptors) for gradient boosting models. Optionally
includes 3D descriptors computed from generated conformers.

Features Computed:
    - Mordred 2D descriptors (~1600): Constitutional, topological, connectivity
    - Mordred 3D descriptors (~200): Geometry, surface area, shape (optional)
    - RDKit LogP: Crippen partition coefficient
    - Solvent one-hot: Binary encoding of solvent type
    - InChIKey hash features: Hashed scaffold identifiers for grouping

Arguments:
    --input, -i          : Input Parquet/CSV file with molecular data
    --out_parquet        : Output Parquet file path
    --smiles_neutral_col : Neutralized SMILES column (default: smiles_neutral)
    --smiles_original_col: Original SMILES column (default: smiles_original)
    --smiles_source      : Which SMILES to use: neutral or original (default: neutral)
    --keep-smiles        : Keep SMILES column in output
    --solvent_col        : Solvent column for one-hot encoding (default: solvent)
    --inchikey_col       : InChIKey/cluster column (default: InChIKey14)
    --ik14_hash_bins     : Number of hash bins for InChIKey (default: 128)
    --keep_inchikey_as_group : Keep InChIKey column renamed as 'groups'
    --include_3d         : Calculate 3D descriptors (slow)
    --nproc              : Number of CPUs for parallel calculation (default: 1)
    --seed_3d            : Random seed for conformer generation (default: 42)
    --ff                 : Force field for optimization: uff or mmff (default: uff)
    --max_atoms_3d       : Skip 3D for molecules with more atoms (default: 200)
    --max_iters_3d       : Max optimization iterations (default: 200)
    --save_csv           : Also save output as CSV

Usage:
    python make_features_mordred.py \\
        --input train.parquet \\
        --out_parquet train_mordred_featured.parquet \\
        --smiles_source neutral \\
        --keep-smiles \\
        --inchikey_col cluster_ecfp4_0p7 \\
        --ik14_hash_bins 128 \\
        --nproc 4 \\
        --include_3d \\
        --ff uff

Output:
    Parquet file with original columns plus:
    - mordred__* columns (~1600-1800 descriptors)
    - rdkit__logP (Crippen LogP)
    - solv_* columns (solvent one-hot)
    - ik14h_* columns (InChIKey hash features)
    - ik14_freq (InChIKey frequency in dataset)

Notes:
    - Supports both 'mordred' and 'mordredcommunity' packages
    - 3D calculation is slow; consider disabling for large datasets
    - Invalid molecules produce NaN values for all descriptors
    - Columns are prefixed to avoid naming collisions
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen

# Try importing Mordred (original or community fork)
try:
    from mordred import Calculator, descriptors
    _MORDRED_PKG = "mordred"
except ImportError:
    try:
        from mordredcommunity import Calculator, descriptors
        _MORDRED_PKG = "mordredcommunity"
    except ImportError:
        sys.exit("[ERROR] Neither 'mordred' nor 'mordredcommunity' is installed.")


# ============================================================================
# MOLECULE UTILITIES
# ============================================================================

def smiles_to_mol(smiles: str):
    """
    Convert SMILES string to RDKit Mol object.
    
    Args:
        smiles: SMILES string
    
    Returns:
        RDKit Mol object or None if conversion fails
    """
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def add_3d_conformer(
    mol,
    seed: int = 42,
    max_iters: int = 200,
    forcefield: str = "uff",
    max_atoms: int = 200
):
    """
    Generate and optimize 3D conformer for a molecule.
    
    Uses ETKDG algorithm for initial embedding and force field optimization
    for geometry refinement.
    
    Args:
        mol: RDKit Mol object
        seed: Random seed for conformer generation
        max_iters: Maximum optimization iterations
        forcefield: Force field to use (uff or mmff)
        max_atoms: Skip molecules exceeding this atom count
    
    Returns:
        Mol object with 3D coordinates or None if generation fails
    """
    if mol is None:
        return None
    
    try:
        if mol.GetNumAtoms() >= max_atoms:
            return None
        
        # Add hydrogens for proper 3D generation
        m = Chem.AddHs(mol)
        
        # ETKDG conformer generation
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        
        if AllChem.EmbedMolecule(m, params) == -1:
            return None
        
        # Force field optimization
        try:
            if forcefield.lower() == "mmff" and AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=int(max_iters))
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=int(max_iters))
        except Exception:
            pass  # Continue with unoptimized geometry
        
        return Chem.RemoveHs(m)
    
    except Exception:
        return None


# ============================================================================
# DESCRIPTOR CALCULATION
# ============================================================================

def compute_mordred(
    df_mols: pd.DataFrame,
    ignore_3d: bool = True,
    nproc: int = 1
) -> pd.DataFrame:
    """
    Calculate Mordred descriptors for a set of molecules.
    
    Args:
        df_mols: DataFrame with 'mol' column containing RDKit Mol objects
        ignore_3d: If True, skip 3D descriptors
        nproc: Number of CPUs for parallel calculation
    
    Returns:
        DataFrame with Mordred descriptor columns (prefixed with 'mordred__')
    """
    print(f"[Mordred] Calculating descriptors using {_MORDRED_PKG} (CPUs={nproc})...")
    
    calc = Calculator(descriptors, ignore_3D=ignore_3d)
    mord = calc.pandas(df_mols["mol"], nproc=nproc)
    
    # Keep only numeric columns and handle infinities
    mord = mord.select_dtypes(include=[np.number])
    mord = mord.replace([np.inf, -np.inf], np.nan)
    
    # Prefix columns to avoid collisions
    mord.columns = [f"mordred__{c}" for c in mord.columns]
    
    return mord


def compute_rdkit_logp(mols: pd.Series) -> pd.Series:
    """
    Calculate Crippen LogP for a series of molecules.
    
    Args:
        mols: Series of RDKit Mol objects
    
    Returns:
        Series of LogP values
    """
    def calc_logp(mol):
        if mol is None:
            return np.nan
        try:
            return float(Crippen.MolLogP(mol))
        except Exception:
            return np.nan
    
    result = mols.apply(calc_logp).astype("float32")
    result.name = "rdkit__logP"
    return result


# ============================================================================
# INCHIKEY FEATURES
# ============================================================================

def stable_hash_to_bin(s: str, n_bins: int) -> int:
    """
    Hash a string to a bin index using SHA-1.
    
    Args:
        s: String to hash
        n_bins: Number of bins
    
    Returns:
        Bin index (0 to n_bins-1)
    """
    h = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)
    return h % n_bins


def inchikey14_hash_features(series: pd.Series, n_bins: int) -> pd.DataFrame:
    """
    Create one-hot hash features from InChIKey first 14 characters.
    
    Args:
        series: Series of InChIKey strings
        n_bins: Number of hash bins
    
    Returns:
        DataFrame with one-hot encoded hash features
    """
    n = len(series)
    arr = np.zeros((n, n_bins), dtype=np.uint8)
    
    idx = series.fillna("").astype(str).apply(
        lambda x: stable_hash_to_bin(x, n_bins)
    ).to_numpy()
    
    arr[np.arange(n), idx] = 1
    
    cols = [f"ik14h_{i:03d}" for i in range(n_bins)]
    return pd.DataFrame(arr, index=series.index, columns=cols)


def inchikey14_frequency(series: pd.Series) -> pd.Series:
    """
    Calculate frequency of each InChIKey in the dataset.
    
    Args:
        series: Series of InChIKey strings
    
    Returns:
        Series of frequencies (0 to 1)
    """
    vc = series.value_counts(dropna=False)
    freq = series.map(vc).astype("float32") / float(len(series))
    return freq.rename("ik14_freq")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for Mordred descriptor calculation."""
    
    ap = argparse.ArgumentParser(
        description="Calculate Mordred molecular descriptors."
    )
    
    # Input/Output
    ap.add_argument("-i", "--input", required=True,
                    help="Input Parquet/CSV file")
    ap.add_argument("--out_parquet", default="features_mordred.parquet",
                    help="Output Parquet file")
    ap.add_argument("--save_csv", action="store_true",
                    help="Also save CSV output")
    
    # SMILES configuration
    ap.add_argument("--smiles_neutral_col", default="smiles_neutral",
                    help="Neutralized SMILES column")
    ap.add_argument("--smiles_original_col", default="smiles_original",
                    help="Original SMILES column")
    ap.add_argument("--smiles_source", choices=["neutral", "original"],
                    default="neutral", help="Which SMILES column to use")
    ap.add_argument("--keep-smiles", action="store_true",
                    help="Keep SMILES column in output")
    
    # Feature columns
    ap.add_argument("--solvent_col", default="solvent",
                    help="Solvent column for one-hot encoding")
    ap.add_argument("--inchikey_col", default="InChIKey14",
                    help="InChIKey/cluster column")
    ap.add_argument("--ik14_hash_bins", type=int, default=128,
                    help="Number of hash bins for InChIKey")
    ap.add_argument("--keep_inchikey_as_group", action="store_true",
                    help="Keep InChIKey column as 'groups'")
    
    # 3D configuration
    ap.add_argument("--include_3d", action="store_true",
                    help="Calculate 3D descriptors (slow)")
    ap.add_argument("--nproc", type=int, default=1,
                    help="Number of CPUs")
    ap.add_argument("--seed_3d", type=int, default=42,
                    help="Random seed for 3D generation")
    ap.add_argument("--ff", choices=["uff", "mmff"], default="uff",
                    help="Force field for optimization")
    ap.add_argument("--max_atoms_3d", type=int, default=200,
                    help="Skip 3D for molecules with more atoms")
    ap.add_argument("--max_iters_3d", type=int, default=200,
                    help="Max optimization iterations")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"[Mordred] Loading {args.input}...")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    print(f"[Mordred] Loaded {len(df):,} molecules")
    
    # Select SMILES column
    smi_col = args.smiles_neutral_col if args.smiles_source == "neutral" else args.smiles_original_col
    if smi_col not in df.columns:
        sys.exit(f"[ERROR] SMILES column '{smi_col}' not found.")
    
    # -------------------------------------------------------------------------
    # Convert to Mol Objects
    # -------------------------------------------------------------------------
    print("[Mordred] Converting SMILES to RDKit molecules...")
    mols = df[smi_col].apply(smiles_to_mol)
    
    # -------------------------------------------------------------------------
    # 3D Conformer Generation (Optional)
    # -------------------------------------------------------------------------
    if args.include_3d:
        print("[Mordred] Generating 3D conformers (this may take a while)...")
        
        def add_3d(mol):
            return add_3d_conformer(
                mol,
                seed=args.seed_3d,
                max_iters=args.max_iters_3d,
                forcefield=args.ff,
                max_atoms=args.max_atoms_3d
            )
        
        mols_for_calc = mols.apply(add_3d)
    else:
        mols_for_calc = mols
    
    # -------------------------------------------------------------------------
    # Calculate RDKit LogP
    # -------------------------------------------------------------------------
    print("[Mordred] Calculating RDKit LogP...")
    rdkit_logp = compute_rdkit_logp(mols)
    
    # -------------------------------------------------------------------------
    # Calculate Mordred Descriptors
    # -------------------------------------------------------------------------
    print("[Mordred] Calculating Mordred descriptors...")
    
    # Filter valid molecules
    valid_mask = mols_for_calc.notna()
    df_mols_valid = pd.DataFrame({"mol": mols_for_calc[valid_mask]})
    
    if len(df_mols_valid) > 0:
        mord_valid = compute_mordred(
            df_mols_valid,
            ignore_3d=not args.include_3d,
            nproc=args.nproc
        )
        # Reindex to original shape
        mord = mord_valid.reindex(df.index)
    else:
        print("[WARN] No valid molecules for Mordred calculation.")
        mord = pd.DataFrame(index=df.index)
    
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"[WARN] {n_invalid:,} molecules failed descriptor calculation")
    
    # -------------------------------------------------------------------------
    # Solvent One-Hot Encoding
    # -------------------------------------------------------------------------
    if args.solvent_col in df.columns:
        print("[Mordred] One-hot encoding solvents...")
        solvent_ohe = pd.get_dummies(
            df[args.solvent_col].astype(str),
            prefix="solv",
            dummy_na=False
        ).astype("int8")
    else:
        solvent_ohe = pd.DataFrame(index=df.index)
    
    # -------------------------------------------------------------------------
    # InChIKey Features
    # -------------------------------------------------------------------------
    ik_col = args.inchikey_col
    if ik_col in df.columns:
        print(f"[Mordred] Computing InChIKey hash features ({args.ik14_hash_bins} bins)...")
        ik_series = df[ik_col].astype(str)
        ik_hash = inchikey14_hash_features(ik_series, args.ik14_hash_bins)
        ik_freq = inchikey14_frequency(ik_series)
    else:
        ik_hash = pd.DataFrame(index=df.index)
        ik_freq = pd.Series(index=df.index, dtype="float32", name="ik14_freq")
    
    # -------------------------------------------------------------------------
    # Assemble Output
    # -------------------------------------------------------------------------
    print("[Mordred] Assembling final dataset...")
    
    parts = [df.copy()]
    parts.append(rdkit_logp.to_frame())
    
    if not mord.empty:
        parts.append(mord)
    if not solvent_ohe.empty:
        parts.append(solvent_ohe)
    if not ik_hash.empty:
        parts.append(ik_hash)
    if ik_freq.notna().any():
        parts.append(ik_freq.to_frame())
    
    if args.keep_inchikey_as_group and ik_col in df.columns:
        parts.append(df[ik_col].astype(str).rename("groups"))
    
    final_df = pd.concat(parts, axis=1)
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[Mordred] Saving to {args.out_parquet}...")
    final_df.to_parquet(args.out_parquet, index=False)
    
    if args.save_csv:
        csv_out = Path(args.out_parquet).with_suffix(".csv")
        final_df.to_csv(csv_out, index=False)
        print(f"[Mordred] Saved CSV: {csv_out}")
    
    # Summary
    n_mordred = len([c for c in final_df.columns if c.startswith("mordred__")])
    print(f"\n[Mordred] Summary:")
    print(f"          Molecules: {len(final_df):,}")
    print(f"          Mordred descriptors: {n_mordred}")
    print(f"          Total columns: {len(final_df.columns):,}")


if __name__ == "__main__":
    main()
