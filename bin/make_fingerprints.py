#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fingerprints.py
--------------------
Generates Morgan Fingerprints (ECFP4) as bit columns.
Performs Butina Clustering based on Tanimoto similarity.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

def to_mol(s: str):
    """Safely converts SMILES to Mol."""
    if pd.isna(s): return None
    try:
        return Chem.MolFromSmiles(str(s))
    except:
        return None

def cluster_fingerprints(bitvects, cutoff: float):
    """
    Performs Butina clustering on a list of RDKit ExplicitBitVects.
    Returns an array of cluster IDs (-1 for errors).
    """
    n = len(bitvects)
    if n == 0: return np.array([], dtype=int)

    print(f"[Fingerprints] Calculating distance matrix for {n} molecules...")
    
    # Calculate condensed distance matrix (Upper Triangle)
    # Memory usage grows N^2. For >50k molecules, this might OOM on small machines.
    dists = []
    
    # Optimized bulk calculation
    for i in tqdm(range(1, n), desc="Similarity Matrix"):
        # BulkTanimoto is much faster than python loops
        sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[:i])
        # Convert similarity to distance (1 - sim)
        dists.extend([1.0 - s for s in sims])

    print(f"[Fingerprints] Clustering (Butina, cutoff={cutoff})...")
    # Run clustering
    clusters = Butina.ClusterData(dists, nPts=n, distThresh=1.0 - cutoff, isDistData=True)
    
    # Assign labels
    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
            
    return labels

def main():
    ap = argparse.ArgumentParser(description="Generate ECFP4 bits and Butina Clusters.")
    ap.add_argument("--input", "-i", required=True, help="Input Parquet/CSV")
    ap.add_argument("--out-parquet", required=True, help="Output Parquet")
    ap.add_argument("--smiles-col", default="smiles_neutral", help="SMILES column name")
    
    # Fingerprint params
    ap.add_argument("--n-bits", type=int, default=2048, help="Fingerprint length (default: 2048)")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (2 = ECFP4)")
    ap.add_argument("--bits-prefix", default="ecfp4_b", help="Prefix for bit columns")
    
    # Clustering params
    ap.add_argument("--cluster-cutoff", type=float, default=0.7, help="Tanimoto cutoff (1-dist) for clustering")
    
    ap.add_argument("--save-csv", action="store_true", help="Also save CSV")
    
    args = ap.parse_args()

    # 1. Load Data
    print(f"[Fingerprints] Loading {args.input}...")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    if args.smiles_col not in df.columns:
        sys.exit(f"[ERROR] Column '{args.smiles_col}' not found.")

    # 2. Generate Fingerprints
    print("[Fingerprints] Generating Morgan Fingerprints...")
    mols = df[args.smiles_col].apply(to_mol)
    
    # Identify valid molecules
    valid_mask = mols.notna()
    valid_idx = np.where(valid_mask)[0]
    mols_valid = mols.iloc[valid_idx]
    
    bitvects_valid = []
    for m in mols_valid:
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=args.radius, nBits=args.n_bits)
        bitvects_valid.append(bv)

    # 3. Expand Bits to Columns (for GBM models)
    # We create a numpy matrix first for speed
    print("[Fingerprints] Expanding bits to columns...")
    n_valid = len(bitvects_valid)
    bits_matrix = np.zeros((n_valid, args.n_bits), dtype=np.uint8)
    
    for i, bv in enumerate(bitvects_valid):
        # RDKit's fast conversion to numpy
        DataStructs.ConvertToNumpyArray(bv, bits_matrix[i])

    # Create DataFrame for bits (Sparse-ish)
    bit_cols = [f"{args.bits_prefix}{i}" for i in range(args.n_bits)]
    
    # Full size matrix (including invalid rows as 0s)
    full_bits_matrix = np.zeros((len(df), args.n_bits), dtype=np.uint8)
    full_bits_matrix[valid_idx] = bits_matrix
    
    df_bits = pd.DataFrame(full_bits_matrix, columns=bit_cols, index=df.index)

    # 4. Perform Clustering (Only on valid molecules)
    labels_valid = cluster_fingerprints(bitvects_valid, cutoff=args.cluster_cutoff)
    
    # Map back to full length
    cluster_full = np.full(len(df), -1, dtype=int)
    cluster_full[valid_idx] = labels_valid

    # Add Cluster Column
    # e.g. cluster_ecfp4_0p7
    cutoff_str = str(args.cluster_cutoff).replace(".", "p")
    cluster_col = f"cluster_ecfp4_{cutoff_str}"
    
    # 5. Merge and Save
    out = pd.concat([df, df_bits], axis=1)
    out[cluster_col] = cluster_full

    # Cleanup legacy columns if they exist
    if "ecfp4_bits" in out.columns:
        out.drop(columns=["ecfp4_bits"], inplace=True)

    print(f"[Fingerprints] Saving to {args.out_parquet}...")
    out.to_parquet(args.out_parquet, index=False)

    if args.save_csv:
        csv_path = Path(args.out_parquet).with_suffix(".csv")
        out.to_csv(csv_path, index=False)
        print(f"[Fingerprints] Saved CSV to {csv_path}")

if __name__ == "__main__":
    main()