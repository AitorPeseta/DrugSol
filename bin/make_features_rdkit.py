#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Features RDKit: Basic Physicochemical Descriptor Calculation
==================================================================

Calculates fundamental physicochemical descriptors using RDKit. These features
are used by the Physics baseline model and as auxiliary features for Graph
Neural Networks.

Features Computed:
    - rdkit__TPSA: Topological Polar Surface Area (Å²)
    - rdkit__logP: Crippen LogP (octanol-water partition coefficient)
    - rdkit__MW: Molecular Weight (Da)

These properties are fundamental indicators of aqueous solubility:
    - TPSA correlates positively with solubility (polar molecules dissolve better)
    - LogP correlates negatively with solubility (lipophilic molecules dissolve worse)
    - MW generally correlates negatively with solubility (larger molecules less soluble)

Arguments:
    --input      : Input Parquet/CSV file with molecular data
    --out        : Output Parquet file path
    --smiles-col : Column name containing SMILES strings (default: smiles_neutral)

Usage:
    python make_features_rdkit.py \\
        --input train.parquet \\
        --out train_rdkit_featured.parquet \\
        --smiles-col smiles_neutral

Output:
    Parquet file with original columns plus:
    - rdkit__TPSA (float32)
    - rdkit__logP (float32)
    - rdkit__MW (float32)
    
    Invalid SMILES produce NaN values for all descriptors.

Notes:
    - Calculation is fast (~10,000 molecules/second)
    - Memory efficient: processes molecules individually
    - Column names prefixed with 'rdkit__' to avoid collisions
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# ============================================================================
# I/O UTILITIES
# ============================================================================

def read_any(path: str) -> pd.DataFrame:
    """
    Read input file as DataFrame (supports Parquet, CSV, TSV).
    
    Args:
        path: File path
    
    Returns:
        Pandas DataFrame
    """
    p = Path(path)
    suffix = p.suffix.lower()
    
    if suffix == ".parquet":
        return pd.read_parquet(p)
    elif suffix in (".csv", ".txt"):
        return pd.read_csv(p)
    elif suffix == ".tsv":
        return pd.read_csv(p, sep='\t')
    else:
        sys.exit(f"[ERROR] Unsupported file format: {suffix}")


# ============================================================================
# DESCRIPTOR CALCULATION
# ============================================================================

def compute_rdkit_basic(mol) -> dict:
    """
    Calculate basic physicochemical properties for a molecule.
    
    Args:
        mol: RDKit Mol object (or None)
    
    Returns:
        Dictionary with TPSA, logP, and MW values (NaN if mol is None)
    """
    if mol is None:
        return {
            "rdkit__TPSA": np.nan,
            "rdkit__logP": np.nan,
            "rdkit__MW": np.nan,
        }
    
    try:
        return {
            "rdkit__TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "rdkit__logP": float(Descriptors.MolLogP(mol)),
            "rdkit__MW": float(Descriptors.MolWt(mol)),
        }
    except Exception:
        return {
            "rdkit__TPSA": np.nan,
            "rdkit__logP": np.nan,
            "rdkit__MW": np.nan,
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for RDKit descriptor calculation."""
    
    ap = argparse.ArgumentParser(
        description="Calculate basic RDKit physicochemical descriptors."
    )
    ap.add_argument("--input", required=True,
                    help="Input Parquet/CSV file")
    ap.add_argument("--out", required=True,
                    help="Output Parquet file")
    ap.add_argument("--smiles-col", default="smiles_neutral",
                    help="SMILES column name (default: smiles_neutral)")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"[RDKit] Loading {args.input}...")
    df = read_any(args.input)
    
    if args.smiles_col not in df.columns:
        sys.exit(f"[ERROR] SMILES column '{args.smiles_col}' not found.")
    
    print(f"[RDKit] Loaded {len(df):,} molecules")
    
    # -------------------------------------------------------------------------
    # Calculate Descriptors
    # -------------------------------------------------------------------------
    print("[RDKit] Computing physicochemical descriptors...")
    
    # Convert SMILES to Mol objects
    mols = df[args.smiles_col].apply(
        lambda x: Chem.MolFromSmiles(str(x)) if pd.notna(x) else None
    )
    
    # Calculate descriptors
    descriptors_list = mols.apply(compute_rdkit_basic).tolist()
    df_desc = pd.DataFrame(descriptors_list, index=df.index)
    
    # Count failures
    n_invalid = df_desc["rdkit__MW"].isna().sum()
    if n_invalid > 0:
        print(f"[WARN] {n_invalid:,} molecules failed descriptor calculation")
    
    # -------------------------------------------------------------------------
    # Merge and Save
    # -------------------------------------------------------------------------
    df_final = pd.concat([df, df_desc], axis=1)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[RDKit] Saving to {out_path}...")
    df_final.to_parquet(out_path, index=False)
    
    # Summary
    print(f"\n[RDKit] Summary:")
    print(f"        Molecules: {len(df_final):,}")
    print(f"        Valid: {len(df_final) - n_invalid:,}")
    print(f"        Features added: rdkit__TPSA, rdkit__logP, rdkit__MW")
    print("[RDKit] Done.")


if __name__ == "__main__":
    main()
