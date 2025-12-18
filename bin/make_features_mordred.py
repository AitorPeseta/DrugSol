#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features_mordred.py
------------------------
Calculates molecular descriptors:
1. RDKit LogP (Crippen)
2. Mordred Descriptors (2D and optionally 3D)
3. Solvent One-Hot Encoding
4. InChIKey Hashing (for grouping features)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen

# Try importing Mordred (original or community fork)
try:
    from mordred import Calculator, descriptors
    _MORDRED_PKG = "mordred"
except ImportError:
    try:
        from mordredcommunity import Calculator, descriptors
        _MORDRED_PKG = "mordredcommunity"
    except ImportError:
        sys.exit("[ERROR] Neither 'mordred' nor 'mordredcommunity' installed.")

# ============================================================================
# UTILS
# ============================================================================

def smiles_to_mol(s):
    if pd.isna(s): return None
    try:
        return Chem.MolFromSmiles(str(s))
    except:
        return None

def add_3d_conformer(mol, seed=42, max_iters=200, forcefield="uff", max_atoms=200):
    """Generates a 3D conformer for the molecule."""
    if mol is None: return None
    try:
        if mol.GetNumAtoms() >= max_atoms: return None
        
        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        
        if AllChem.EmbedMolecule(m, params) == -1:
            return None
            
        try:
            if forcefield.lower() == "mmff" and AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=int(max_iters))
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=int(max_iters))
        except:
            pass # Optimization failed, return unoptimized geometry
            
        return Chem.RemoveHs(m)
    except:
        return None

def compute_mordred(df_mols, ignore_3d=True, nproc=1):
    """Runs the Mordred calculator."""
    print(f"[Mordred] Calculating descriptors using package: {_MORDRED_PKG} (CPUs={nproc})...")
    calc = Calculator(descriptors, ignore_3D=ignore_3d)
    
    # calc.pandas returns a DataFrame with descriptors
    # IMPORTANT: Ensure no None values are passed here
    mord = calc.pandas(df_mols["mol"], nproc=nproc)
    
    # Clean up: Keep only numbers, remove Infs
    mord = mord.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    
    # Prefix columns to avoid collisions
    mord.columns = [f"mordred__{c}" for c in mord.columns]
    return mord

def compute_rdkit_logp(mols: pd.Series) -> pd.Series:
    """Calculates Crippen LogP."""
    def _lp(m):
        if m is None: return np.nan
        try:
            return float(Crippen.MolLogP(m))
        except:
            return np.nan
            
    s = mols.apply(_lp).astype("float32")
    s.name = "rdkit__logP"
    return s

def _stable_hash_to_bin(s: str, n_bins: int) -> int:
    import hashlib
    h = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)
    return h % n_bins

def inchikey14_hash_features(series: pd.Series, n_bins: int) -> pd.DataFrame:
    """Creates hashing features from InChIKey first 14 chars."""
    n = len(series)
    arr = np.zeros((n, n_bins), dtype=np.uint8)
    idx = series.fillna("").astype(str).apply(lambda x: _stable_hash_to_bin(x, n_bins)).to_numpy()
    arr[np.arange(n), idx] = 1
    cols = [f"ik14h_{i:03d}" for i in range(n_bins)]
    return pd.DataFrame(arr, index=series.index, columns=cols)

def inchikey14_frequency(series: pd.Series) -> pd.Series:
    vc = series.value_counts(dropna=False)
    freq = series.map(vc).astype("float32") / float(len(series))
    return freq.astype("float32").rename("ik14_freq")

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Calculate Molecular Descriptors (Mordred, LogP, Solvent, Hash).")
    
    # Input/Output
    ap.add_argument("-i", "--input", required=True, help="Input file (.parquet/.csv)")
    ap.add_argument("--out_parquet", default="features_mordred.parquet", help="Output file")
    ap.add_argument("--save_csv", action="store_true", help="Save CSV copy")
    
    # Columns
    ap.add_argument("--smiles_neutral_col", default="smiles_neutral")
    ap.add_argument("--smiles_original_col", default="smiles_original")
    ap.add_argument("--smiles_source", choices=["neutral", "original"], default="neutral", help="Which SMILES col to use")
    ap.add_argument("--keep-smiles", action="store_true", help="Keep SMILES column in output")
    
    ap.add_argument("--solvent_col", default="solvent")
    ap.add_argument("--inchikey_col", default="InChIKey14")
    
    # Hashing
    ap.add_argument("--ik14_hash_bins", type=int, default=128)
    ap.add_argument("--keep_inchikey_as_group", action="store_true", help="Keep InChIKey column renamed as 'groups'")
    
    # 3D & Mordred Settings
    ap.add_argument("--include_3d", action="store_true", help="Calculate 3D descriptors (Slow!)")
    ap.add_argument("--nproc", type=int, default=1, help="Number of CPUs")
    ap.add_argument("--seed_3d", type=int, default=42)
    ap.add_argument("--ff", choices=["uff", "mmff"], default="uff")
    ap.add_argument("--max_atoms_3d", type=int, default=200)
    ap.add_argument("--max_iters_3d", type=int, default=200)
    
    args = ap.parse_args()

    # 1. Load Data
    print(f"[Mordred] Loading {args.input}...")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # Select SMILES column
    smi_col = args.smiles_neutral_col if args.smiles_source == "neutral" else args.smiles_original_col
    if smi_col not in df.columns:
        sys.exit(f"[ERROR] SMILES column '{smi_col}' missing.")

    # 2. Prepare Molecules
    print("[Mordred] Converting SMILES to RDKit Mols...")
    mols = df[smi_col].apply(smiles_to_mol)
    
    # 3. 3D Conformation (Optional)
    if args.include_3d:
        print("[Mordred] Generating 3D conformers (this is slow)...")
        def _add3d(m):
            if m is None: return None
            return add_3d_conformer(m, seed=args.seed_3d, max_iters=args.max_iters_3d, 
                                    forcefield=args.ff, max_atoms=args.max_atoms_3d)
        mols_for_calc = mols.apply(_add3d)
    else:
        mols_for_calc = mols

    # 4. Calculate RDKit LogP (Works on 2D if 3D fails, so use original mols if needed, but best on processed)
    print("[Mordred] Calculating RDKit LogP...")
    rdkit_logp = compute_rdkit_logp(mols)

    # 5. Calculate Mordred
    print("[Mordred] Calculating Mordred Descriptors...")
    
    # --- FIX START ---
    # Filtrar explícitamente cualquier molécula que sea None.
    # Esto ocurre si la conversión 2D falló O si la generación 3D falló.
    valid_calc_mask = mols_for_calc.notna()
    
    df_mols_valid = pd.DataFrame({"mol": mols_for_calc[valid_calc_mask]})
    
    if len(df_mols_valid) > 0:
        # Calcular solo para las válidas
        mord_valid = compute_mordred(df_mols_valid, ignore_3d=not args.include_3d, nproc=args.nproc)
        # Reindexar para volver al tamaño original (rellena con NaN las filas fallidas)
        mord = mord_valid.reindex(df.index)
    else:
        print("[WARN] No valid molecules for Mordred calculation. Returning empty features.")
        mord = pd.DataFrame(index=df.index)
    # --- FIX END ---

    # 6. Solvent One-Hot Encoding
    print("[Mordred] One-Hot Encoding Solvents...")
    if args.solvent_col in df.columns:
        solvent_ohe = pd.get_dummies(df[args.solvent_col].astype(str), prefix="solv", dummy_na=False).astype("int8")
    else:
        solvent_ohe = pd.DataFrame(index=df.index)

    # 7. InChIKey Features (Hash & Frequency)
    ik_col = args.inchikey_col
    if ik_col in df.columns:
        print(f"[Mordred] Hashing {ik_col}...")
        ik_series = df[ik_col].astype(str)
        ik_hash = inchikey14_hash_features(ik_series, args.ik14_hash_bins)
        ik_freq = inchikey14_frequency(ik_series)
    else:
        ik_hash = pd.DataFrame(index=df.index)
        ik_freq = pd.Series(index=df.index, dtype="float32", name="ik14_freq")

    # 8. Assemble Output
    print("[Mordred] Assembling final dataset...")
    parts = [df.copy()] # Keep original columns
    parts.append(rdkit_logp.to_frame())
    if not mord.empty: parts.append(mord)
    if not solvent_ohe.empty: parts.append(solvent_ohe)
    if not ik_hash.empty: parts.append(ik_hash)
    if ik_freq.notna().any(): parts.append(ik_freq.to_frame())
    
    if args.keep_inchikey_as_group and ik_col in df.columns:
        parts.append(df[ik_col].astype(str).rename("groups"))

    final_df = pd.concat(parts, axis=1)

    # 9. Save
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(args.out_parquet, index=False)
    print(f"[Mordred] Saved Parquet: {args.out_parquet}")

    if args.save_csv:
        csv_out = Path(args.out_parquet).with_suffix(".csv")
        final_df.to_csv(csv_out, index=False)
        print(f"[Mordred] Saved CSV: {csv_out}")

if __name__ == "__main__":
    main()