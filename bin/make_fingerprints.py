#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Fingerprints: Morgan Fingerprint Generation and Butina Clustering
=======================================================================

Generates Morgan fingerprints (ECFP4 equivalent) and performs Butina clustering
based on Tanimoto similarity. The cluster assignments enable scaffold-based
train/test splitting to prevent data leakage from structurally similar compounds.

Algorithm Overview:
    1. Parse SMILES strings to RDKit Mol objects
    2. Generate Morgan fingerprints (circular fingerprints)
    3. Expand fingerprint to individual bit columns (for tree-based models)
    4. Calculate condensed Tanimoto distance matrix
    5. Perform Butina (Taylor-Butina) clustering
    6. Assign cluster IDs to each molecule

Arguments:
    --input, -i        : Input Parquet/CSV file with molecular data
    --out-parquet      : Output Parquet file path
    --smiles-col       : Column name containing SMILES strings (default: smiles_neutral)
    --n-bits           : Fingerprint vector length (default: 2048)
    --radius           : Morgan radius (default: 2, equivalent to ECFP4)
    --bits-prefix      : Prefix for bit column names (default: ecfp4_b)
    --cluster-cutoff   : Tanimoto similarity threshold for clustering (default: 0.7)
    --save-csv         : Also save output as CSV file

Usage:
    python make_fingerprints.py \\
        --input engineered_features.parquet \\
        --out-parquet cluster_ecfp4_0p7_fingerprint.parquet \\
        --smiles-col smiles_neutral \\
        --n-bits 2048 \\
        --radius 2 \\
        --cluster-cutoff 0.7 \\
        --save-csv

Output:
    Parquet file with:
    - Original columns preserved
    - Fingerprint bit columns: {prefix}0, {prefix}1, ..., {prefix}{n_bits-1}
    - Cluster assignment column: cluster_ecfp4_{cutoff} (e.g., cluster_ecfp4_0p7)

Notes:
    - Memory usage scales O(N²) for clustering distance matrix
    - For datasets >50k molecules, consider chunked processing or approximate methods
    - Invalid SMILES receive cluster ID = -1 and all-zero fingerprint bits
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# RDKit imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def to_mol(smiles: str):
    """
    Safely convert SMILES string to RDKit Mol object.
    
    Args:
        smiles: SMILES string representation of molecule
    
    Returns:
        RDKit Mol object or None if conversion fails
    """
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def cluster_fingerprints(bitvects: list, cutoff: float) -> np.ndarray:
    """
    Perform Butina clustering on molecular fingerprints.
    
    The Butina algorithm (Taylor-Butina) is a leader-based clustering method
    that groups molecules based on Tanimoto similarity. It's efficient for
    large datasets and produces non-overlapping clusters.
    
    Algorithm:
    1. Calculate pairwise Tanimoto distances (1 - similarity)
    2. Sort molecules by number of neighbors within cutoff
    3. Assign cluster centroids (leaders) starting from most connected
    4. Assign remaining molecules to nearest centroid
    
    Args:
        bitvects: List of RDKit ExplicitBitVect fingerprints
        cutoff: Tanimoto similarity threshold (0.0-1.0)
                Molecules with similarity >= cutoff are in same cluster
    
    Returns:
        NumPy array of cluster IDs (same length as bitvects)
        Invalid molecules receive cluster ID = -1
    
    Memory Warning:
        Distance matrix size is O(N²). For 50k molecules:
        - ~1.25 billion distances
        - ~10 GB memory for float64
        Consider chunked processing for very large datasets.
    """
    n = len(bitvects)
    if n == 0:
        return np.array([], dtype=int)
    
    print(f"[Fingerprints] Calculating distance matrix for {n:,} molecules...")
    
    # Calculate condensed distance matrix (upper triangle only)
    # Using BulkTanimotoSimilarity for efficiency
    dists = []
    
    for i in tqdm(range(1, n), desc="Similarity Matrix"):
        # BulkTanimoto calculates similarity to all previous molecules at once
        sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[:i])
        # Convert similarity to distance: d = 1 - similarity
        dists.extend([1.0 - s for s in sims])
    
    print(f"[Fingerprints] Clustering with Butina (cutoff={cutoff})...")
    
    # Run Butina clustering
    # distThresh is the distance threshold: 1 - similarity_cutoff
    clusters = Butina.ClusterData(
        dists,
        nPts=n,
        distThresh=1.0 - cutoff,
        isDistData=True
    )
    
    # Assign cluster labels
    labels = np.full(n, -1, dtype=int)
    for cluster_id, members in enumerate(clusters):
        for member_idx in members:
            labels[member_idx] = cluster_id
    
    n_clusters = len(clusters)
    print(f"[Fingerprints] Found {n_clusters:,} clusters")
    
    return labels


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for fingerprint generation script."""
    
    ap = argparse.ArgumentParser(
        description="Generate ECFP4 fingerprints and perform Butina clustering."
    )
    
    # Input/Output
    ap.add_argument("--input", "-i", required=True,
                    help="Input Parquet/CSV file")
    ap.add_argument("--out-parquet", required=True,
                    help="Output Parquet file path")
    ap.add_argument("--smiles-col", default="smiles_neutral",
                    help="SMILES column name (default: smiles_neutral)")
    
    # Fingerprint parameters
    ap.add_argument("--n-bits", type=int, default=2048,
                    help="Fingerprint vector length (default: 2048)")
    ap.add_argument("--radius", type=int, default=2,
                    help="Morgan radius (default: 2 = ECFP4)")
    ap.add_argument("--bits-prefix", default="ecfp4_b",
                    help="Prefix for bit column names (default: ecfp4_b)")
    
    # Clustering parameters
    ap.add_argument("--cluster-cutoff", type=float, default=0.7,
                    help="Tanimoto similarity cutoff (default: 0.7)")
    
    # Output options
    ap.add_argument("--save-csv", action="store_true",
                    help="Also save output as CSV file")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Input Data
    # -------------------------------------------------------------------------
    print(f"[Fingerprints] Loading {args.input}...")
    
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    if args.smiles_col not in df.columns:
        sys.exit(f"[ERROR] Column '{args.smiles_col}' not found in input file.")
    
    print(f"[Fingerprints] Loaded {len(df):,} rows")
    
    # -------------------------------------------------------------------------
    # Generate Fingerprints
    # -------------------------------------------------------------------------
    print("[Fingerprints] Generating Morgan fingerprints...")
    
    mols = df[args.smiles_col].apply(to_mol)
    
    # Identify valid molecules
    valid_mask = mols.notna()
    valid_idx = np.where(valid_mask)[0]
    mols_valid = mols.iloc[valid_idx]
    
    n_valid = len(mols_valid)
    n_invalid = len(df) - n_valid
    
    if n_invalid > 0:
        print(f"[WARN] {n_invalid:,} molecules failed to parse ({n_invalid/len(df)*100:.1f}%)")
    
    # Generate fingerprints for valid molecules
    bitvects_valid = []
    for mol in mols_valid:
        bv = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=args.radius,
            nBits=args.n_bits
        )
        bitvects_valid.append(bv)
    
    # -------------------------------------------------------------------------
    # Expand Bits to Columns
    # -------------------------------------------------------------------------
    print("[Fingerprints] Expanding bits to columns...")
    
    # Create numpy matrix for efficiency
    bits_matrix = np.zeros((n_valid, args.n_bits), dtype=np.uint8)
    
    for i, bv in enumerate(bitvects_valid):
        DataStructs.ConvertToNumpyArray(bv, bits_matrix[i])
    
    # Create full-size matrix (invalid rows remain zeros)
    full_bits_matrix = np.zeros((len(df), args.n_bits), dtype=np.uint8)
    full_bits_matrix[valid_idx] = bits_matrix
    
    # Create DataFrame with bit columns
    bit_cols = [f"{args.bits_prefix}{i}" for i in range(args.n_bits)]
    df_bits = pd.DataFrame(full_bits_matrix, columns=bit_cols, index=df.index)
    
    # -------------------------------------------------------------------------
    # Perform Butina Clustering
    # -------------------------------------------------------------------------
    labels_valid = cluster_fingerprints(bitvects_valid, cutoff=args.cluster_cutoff)
    
    # Map cluster labels back to full DataFrame
    cluster_full = np.full(len(df), -1, dtype=int)
    cluster_full[valid_idx] = labels_valid
    
    # Generate cluster column name (e.g., "cluster_ecfp4_0p7")
    cutoff_str = str(args.cluster_cutoff).replace(".", "p")
    cluster_col = f"cluster_ecfp4_{cutoff_str}"
    
    # -------------------------------------------------------------------------
    # Merge and Save
    # -------------------------------------------------------------------------
    print("[Fingerprints] Merging results...")
    
    out = pd.concat([df, df_bits], axis=1)
    out[cluster_col] = cluster_full
    
    # Remove legacy columns if present
    if "ecfp4_bits" in out.columns:
        out.drop(columns=["ecfp4_bits"], inplace=True)
    
    print(f"[Fingerprints] Saving to {args.out_parquet}...")
    out.to_parquet(args.out_parquet, index=False)
    
    if args.save_csv:
        csv_path = Path(args.out_parquet).with_suffix(".csv")
        out.to_csv(csv_path, index=False)
        print(f"[Fingerprints] Saved CSV to {csv_path}")
    
    # Print summary statistics
    n_clusters = len(set(cluster_full)) - (1 if -1 in cluster_full else 0)
    print(f"[Fingerprints] Summary:")
    print(f"   -> Total molecules: {len(df):,}")
    print(f"   -> Valid molecules: {n_valid:,}")
    print(f"   -> Fingerprint bits: {args.n_bits}")
    print(f"   -> Clusters found: {n_clusters:,}")
    print(f"   -> Output columns: {len(out.columns):,}")


if __name__ == "__main__":
    main()
