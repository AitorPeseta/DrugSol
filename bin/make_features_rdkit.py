#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features_rdkit.py
----------------------
Calculates basic RDKit features (TPSA, logP, MW).
Optimized for performance using batch processing.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------- I/O helpers ----------------

def read_any(path: str) -> pd.DataFrame:
    """Reads Parquet or CSV."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(p)
    else:
        raise SystemExit(f"[ERROR] Unsupported format: {p.suffix}")

def compute_rdkit_basic(mol):
    """
    Returns a dictionary with basic physicochemical properties.
    Returns NaNs if mol is None.
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

def main():
    ap = argparse.ArgumentParser(description="Add basic RDKit features.")
    ap.add_argument("--input", required=True, help="Input Parquet/CSV")
    ap.add_argument("--out", required=True, help="Output Parquet filename")
    ap.add_argument("--smiles-col", default="smiles_neutral", help="SMILES column name")
    args = ap.parse_args()

    # 1. Load
    print(f"[RDKit Features] Loading {args.input}...")
    df = read_any(args.input)
    
    if args.smiles_col not in df.columns:
        raise SystemExit(f"[ERROR] Column '{args.smiles_col}' not found.")

    # 2. Calculate Features
    print(f"[RDKit Features] Computing descriptors for {len(df)} molecules...")
    
    # Prepare molecules generator (lazy evaluation)
    # We use a simple apply to convert to Mol objects first
    mols = df[args.smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)) if pd.notna(x) else None)
    
    # Calculate descriptors as a list of dicts (Faster than df.at loop)
    # We add the prefix 'rdkit__' to avoid collisions
    descriptors_list = mols.apply(compute_rdkit_basic).tolist()
    
    # Convert to DataFrame
    df_desc = pd.DataFrame(descriptors_list, index=df.index)
    
    # 3. Merge
    df_final = pd.concat([df, df_desc], axis=1)

    # 4. Save
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[RDKit Features] Saving to {p}...")
    df_final.to_parquet(p, index=False)
    print("[RDKit Features] Done.")

if __name__ == "__main__":
    main()