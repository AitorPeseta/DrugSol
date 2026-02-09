#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
    standardize_smiles.py
========================================================================================
    Standardizes molecular structures and generates unique identifiers.
    
    This is a critical step for ensuring consistent molecular representations
    across the dataset, which is essential for:
    - Accurate duplicate detection (same molecule, different SMILES)
    - Consistent fingerprint generation
    - Reproducible model training
    
    Standardization Pipeline (RDKit MolStandardize):
    1. Parse SMILES string
    2. Remove salts and counterions (keeps only largest fragment)
    3. Normalize functional groups (nitro, sulfo, phospho, etc.)
    4. Reionize to consistent protonation state
    5. Neutralize charges where chemically appropriate
    6. Canonicalize tautomers (optional, can be slow)
    7. Generate canonical SMILES and InChIKey
    
    Deduplication Logic:
    - Groups entries by InChIKey + solvent + temperature
    - If logS values within threshold: compute average (consensus)
    - If logS values conflict (diff > threshold): remove ALL entries (unreliable data)
    - Generates detailed conflict report for QC
    
    Unique ID Generation:
    - Format: {InChIKey}|{solvent}|{temp_C}
    - Ensures each row represents a unique measurement condition
    
    Arguments:
        --in              Input Parquet file with SMILES column
        --out             Output Parquet file path
        --overwrite-inchikey  Regenerate InChIKey even if exists
        --no-tautomer     Skip tautomer canonicalization (faster)
        --engine          Processing engine: 'auto', 'mp' (multiprocess), 'pandas'
        --workers         Number of parallel workers for 'mp' engine (default: 4)
        --chunksize       Chunk size for parallel processing (default: 2000)
        --export-csv      Also export CSV copy
        --dedup           Enable deduplication
        --dedup-thresh    Max logS difference for consensus (default: 0.7)
    
    Usage:
        # Basic standardization with deduplication
        python standardize_smiles.py --in filtered.parquet --out standard.parquet --dedup
        
        # Fast mode (skip tautomers, single process)
        python standardize_smiles.py --in data.parquet --out out.parquet --no-tautomer --engine pandas
        
        # Parallel processing with custom threshold
        python standardize_smiles.py --in data.parquet --out out.parquet \\
            --dedup --dedup-thresh 0.5 --workers 8
    
    Output:
        Parquet file with added columns:
        - smiles_neutral: Standardized canonical SMILES
        - InChIKey: Unique molecular identifier
        - row_uid: Unique row identifier ({InChIKey}|{solvent}|{temp_C})
        
        Also generates conflict report to stdout if deduplication finds issues.
----------------------------------------------------------------------------------------
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# RDKit imports
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

# ======================================================================================
#     CONSTANTS
# ======================================================================================

# Default parameters
DEFAULT_WORKERS = 4
DEFAULT_CHUNKSIZE = 2000
DEFAULT_DEDUP_THRESHOLD = 0.7

# Minimum rows to trigger multiprocessing
MP_THRESHOLD = 5000

# ======================================================================================
#     STANDARDIZER SETUP
# ======================================================================================

# Global standardizer (cached per process for multiprocessing)
_STANDARDIZER = None


def _create_standardizer() -> tuple:
    """
    Create RDKit standardization pipeline components.
    
    Returns:
        Tuple of (normalizer, reionizer, uncharger, tautomer_canonicalizer)
    """
    normalizer = rdMolStandardize.Normalizer()
    reionizer = rdMolStandardize.Reionizer()
    uncharger = rdMolStandardize.Uncharger()
    
    # Handle different RDKit versions
    try:
        taut_can = rdMolStandardize.TautomerCanonicalizer()
        def canonize(mol):
            return taut_can.canonicalize(mol)
    except AttributeError:
        # Older RDKit versions
        te = rdMolStandardize.TautomerEnumerator()
        def canonize(mol):
            return te.Canonicalize(mol)
    
    return normalizer, reionizer, uncharger, canonize


def _get_standardizer() -> tuple:
    """Get or create cached standardizer."""
    global _STANDARDIZER
    if _STANDARDIZER is None:
        _STANDARDIZER = _create_standardizer()
    return _STANDARDIZER

# ======================================================================================
#     SMILES STANDARDIZATION
# ======================================================================================

def standardize_single(smiles: str, do_tautomer: bool) -> Tuple[Optional[str], Optional[str]]:
    """
    Standardize a single SMILES string.
    
    Args:
        smiles: Input SMILES string
        do_tautomer: Whether to canonicalize tautomers
        
    Returns:
        Tuple of (canonical_smiles, inchikey) or (None, None) if invalid
    """
    # Handle null/NA values robustly
    if pd.isna(smiles):
        return (None, None)
    
    # Convert to string and clean
    smiles_str = str(smiles).strip()
    if not smiles_str or smiles_str.lower() == 'nan':
        return (None, None)
    
    # Quick filter: reject salts (contain '.')
    if "." in smiles_str:
        return (None, None)
    
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return (None, None)
    
    # Robust salt filter: reject multi-fragment molecules
    if len(Chem.GetMolFrags(mol)) > 1:
        return (None, None)
    
    # Apply standardization pipeline
    normalizer, reionizer, uncharger, canonize = _get_standardizer()
    
    try:
        mol = normalizer.normalize(mol)
        mol = reionizer.reionize(mol)
        mol = uncharger.uncharge(mol)
        
        if do_tautomer:
            mol = canonize(mol)
        
        Chem.SanitizeMol(mol)
        
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        inchikey = inchi.MolToInchiKey(mol)
        
        return (canonical_smiles, inchikey)
        
    except Exception:
        return (None, None)

# ======================================================================================
#     PARALLEL PROCESSING
# ======================================================================================

def _process_chunk(
    idx_range: Tuple[int, int],
    smiles_list: List[str],
    do_tautomer: bool
) -> Tuple[Tuple[int, int], List[Optional[str]], List[Optional[str]]]:
    """Process a chunk of SMILES in parallel worker."""
    out_smiles = []
    out_inchikeys = []
    
    for smi in smiles_list:
        canonical, ik = standardize_single(smi, do_tautomer)
        out_smiles.append(canonical)
        out_inchikeys.append(ik)
    
    return (idx_range, out_smiles, out_inchikeys)


def standardize_multiprocess(
    df: pd.DataFrame,
    src_col: str,
    do_tautomer: bool,
    workers: int,
    chunksize: int
) -> pd.DataFrame:
    """
    Standardize SMILES using multiprocessing.
    
    Args:
        df: Input DataFrame
        src_col: Source SMILES column name
        do_tautomer: Whether to canonicalize tautomers
        workers: Number of parallel workers
        chunksize: Rows per chunk
        
    Returns:
        DataFrame with smiles_neutral and InChIKey columns added
    """
    n = len(df)
    if n == 0:
        df["smiles_neutral"] = pd.Series(dtype="string")
        df["InChIKey"] = pd.Series(dtype="string")
        return df
    
    # Auto-calculate chunksize if not specified
    if chunksize <= 0:
        chunksize = max(1000, n // max(1, workers * 4))
    
    # Create index ranges for chunks
    ranges = [(start, min(start + chunksize, n)) for start in range(0, n, chunksize)]
    
    # Initialize result arrays
    smiles_neutral = [None] * n
    inchikeys = [None] * n
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for (start, stop) in ranges:
            smiles_chunk = df[src_col].iloc[start:stop].tolist()
            futures.append(
                executor.submit(_process_chunk, (start, stop), smiles_chunk, do_tautomer)
            )
        
        for future in as_completed(futures):
            (start, stop), smi_list, ik_list = future.result()
            smiles_neutral[start:stop] = smi_list
            inchikeys[start:stop] = ik_list
    
    df["smiles_neutral"] = pd.Series(smiles_neutral, dtype="string")
    df["InChIKey"] = pd.Series(inchikeys, dtype="string")
    
    return df


def standardize_pandas(
    df: pd.DataFrame,
    src_col: str,
    do_tautomer: bool
) -> pd.DataFrame:
    """
    Standardize SMILES using pandas apply (single process).
    
    Args:
        df: Input DataFrame
        src_col: Source SMILES column name
        do_tautomer: Whether to canonicalize tautomers
        
    Returns:
        DataFrame with smiles_neutral and InChIKey columns added
    """
    # Apply standardization
    pairs = df[src_col].astype(object).apply(
        lambda s: standardize_single(s, do_tautomer)
    )
    
    df["smiles_neutral"] = pairs.map(lambda t: t[0]).astype("string")
    df["InChIKey"] = pairs.map(lambda t: t[1]).astype("string")
    
    return df

# ======================================================================================
#     DEDUPLICATION
# ======================================================================================

def deduplicate_data(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Deduplicate entries based on InChIKey, solvent, and temperature.
    
    Logic:
    - If logS values are close (within threshold): average them
    - If logS values conflict (diff > threshold): remove ALL entries
    
    Args:
        df: Input DataFrame with InChIKey, solvent, temp_C, logS columns
        threshold: Maximum logS difference to consider values consistent
        
    Returns:
        Deduplicated DataFrame with conflict report printed
    """
    print(f"[Dedup] Starting strict deduplication (threshold={threshold})...")
    
    # Prepare grouping columns
    df = df.dropna(subset=["InChIKey"])
    df["temp_C_grp"] = df["temp_C"].fillna(-9999)
    df["solvent_grp"] = df["solvent"].fillna("UNKNOWN")
    
    # Identify duplicates (mark ALL occurrences)
    dup_mask = df.duplicated(
        subset=["InChIKey", "solvent_grp", "temp_C_grp"],
        keep=False
    )
    
    df_unique = df[~dup_mask].copy()  # Single occurrences (safe)
    df_dups = df[dup_mask].copy()      # Duplicates to resolve
    
    if df_dups.empty:
        print("[Dedup] No duplicates found.")
        cols_to_drop = ["temp_C_grp", "solvent_grp"]
        if "source" in df.columns:
            cols_to_drop.append("source")
        return df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"[Dedup] Analyzing {len(df_dups):,} duplicate rows...")
    
    # Process duplicates group by group
    grouped = df_dups.groupby(["InChIKey", "solvent_grp", "temp_C_grp"])
    
    resolved_rows = []
    conflict_count = 0
    dropped_groups = 0
    
    print("\n" + "=" * 80)
    print(f" CONFLICT REPORT (threshold > {threshold})")
    print("=" * 80)
    
    for name, group in grouped:
        vals = group["logS"].values
        sources = group["source"].values if "source" in group.columns else ["?"] * len(vals)
        smiles = group["smiles_neutral"].iloc[0]
        
        # Calculate median and deviations
        median_val = np.median(vals)
        diffs = np.abs(vals - median_val)
        is_good = diffs <= threshold
        clean_vals = vals[is_good]
        
        # Decision logic
        if len(clean_vals) == 0:
            # SEVERE CONFLICT: All values disagree - remove all
            status_msg = "SEVERE CONFLICT -> ALL REMOVED"
            dropped_groups += 1
            conflict_count += 1
            
        elif len(clean_vals) < len(vals):
            # PARTIAL CONFLICT: Some outliers detected
            final_logS = np.mean(clean_vals)
            status_msg = "OUTLIER DETECTED -> Discordant value removed"
            conflict_count += 1
            
            row = group.iloc[0].copy()
            row["logS"] = final_logS
            resolved_rows.append(row)
            
        else:
            # CONSENSUS: All values agree
            final_logS = np.mean(clean_vals)
            row = group.iloc[0].copy()
            row["logS"] = final_logS
            resolved_rows.append(row)
            continue  # Don't print if all OK
        
        # Print conflict details
        temp_str = f"{name[2]}°C" if name[2] != -9999 else "N/A"
        smiles_short = smiles[:30] + "..." if len(str(smiles)) > 30 else smiles
        print(f"\nMol: {smiles_short} | {name[1]} | {temp_str}")
        print(f"Status: {status_msg}")
        
        for v, s, good in zip(vals, sources, is_good):
            flag = "" if good else " [REMOVED]"
            print(f"   logS: {v:+.4f}  [{s}]{flag}")
    
    print("=" * 80 + "\n")
    
    # Reconstruct DataFrame
    if resolved_rows:
        df_resolved = pd.DataFrame(resolved_rows)
    else:
        df_resolved = pd.DataFrame(columns=df.columns)
    
    # Clean up auxiliary columns
    final_cols = [c for c in df.columns if c not in ["temp_C_grp", "solvent_grp", "source"]]
    
    df_resolved = df_resolved.reindex(columns=final_cols)
    df_unique = df_unique.reindex(columns=final_cols)
    
    df_clean = pd.concat([df_unique, df_resolved], ignore_index=True)
    
    # Summary
    print(f"[Dedup] Summary:")
    print(f"   Unique rows (no duplicates):  {len(df_unique):,}")
    print(f"   Duplicate groups processed:   {len(grouped):,}")
    print(f"   Groups REMOVED (conflict):    {dropped_groups:,}")
    print(f"   Final row count:              {len(df_clean):,}")
    
    return df_clean

# ======================================================================================
#     UNIQUE ID GENERATION
# ======================================================================================

def generate_row_uids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate unique row identifiers.
    
    Format: {InChIKey}|{solvent}|{temp_C}
    
    Args:
        df: DataFrame with InChIKey, solvent, temp_C columns
        
    Returns:
        DataFrame with row_uid column added
    """
    def normalize_str(s):
        return s.astype("string").fillna("UNKNOWN").str.strip()
    
    def normalize_temp(s):
        return pd.to_numeric(s, errors="coerce")
    
    ik = normalize_str(df["InChIKey"])
    solv = normalize_str(df["solvent"])
    temp = normalize_temp(df["temp_C"])
    temp_str = temp.apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NA")
    
    df["row_uid"] = (ik + "|" + solv + "|" + temp_str).astype("string")
    
    # Handle any remaining duplicates
    if df["row_uid"].duplicated().any():
        n_dup = df["row_uid"].duplicated().sum()
        print(f"[WARNING] {n_dup:,} duplicate UIDs found. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["row_uid"], keep="first")
    
    return df

# ======================================================================================
#     ARGUMENT PARSING
# ======================================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standardize SMILES and deduplicate solubility data.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument("--in", dest="inp", required=True, help="Input Parquet file")
    parser.add_argument("--out", dest="out", required=True, help="Output Parquet file")
    parser.add_argument("--export-csv", action="store_true", help="Also export CSV")
    
    # Standardization options
    parser.add_argument(
        "--overwrite-inchikey",
        action="store_true",
        help="Regenerate InChIKey even if it exists"
    )
    parser.add_argument(
        "--no-tautomer",
        action="store_true",
        help="Skip tautomer canonicalization (faster)"
    )
    
    # Processing engine
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "mp", "pandas"],
        help="Processing engine: auto, mp (multiprocess), pandas (default: auto)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help=f"Chunk size for parallel processing (default: {DEFAULT_CHUNKSIZE})"
    )
    
    # Deduplication
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Enable deduplication"
    )
    parser.add_argument(
        "--dedup-thresh",
        type=float,
        default=DEFAULT_DEDUP_THRESHOLD,
        help=f"Max logS difference for consensus (default: {DEFAULT_DEDUP_THRESHOLD})"
    )
    
    return parser.parse_args()

# ======================================================================================
#     MAIN ENTRY POINT
# ======================================================================================

def main() -> None:
    """Main entry point for SMILES standardization."""
    args = parse_args()
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print(f"[Standardize] Loading: {args.inp}")
    df = pd.read_parquet(args.inp)
    initial_rows = len(df)
    print(f"[Standardize] Input rows: {initial_rows:,}")
    
    # -------------------------------------------------------------------------
    # Determine source column
    # -------------------------------------------------------------------------
    src_col = None
    if "smiles_neutral" in df.columns and df["smiles_neutral"].notna().any():
        src_col = "smiles_neutral"
    elif "smiles_original" in df.columns:
        src_col = "smiles_original"
    
    if src_col is None:
        sys.exit("[ERROR] No SMILES column found (smiles_neutral or smiles_original)")
    
    # -------------------------------------------------------------------------
    # Configure processing
    # -------------------------------------------------------------------------
    do_tautomer = not args.no_tautomer
    engine = args.engine
    
    if engine == "auto":
        engine = "mp" if len(df) >= MP_THRESHOLD else "pandas"
    
    print(f"[Standardize] Source column: {src_col}")
    print(f"[Standardize] Engine: {engine}, Tautomer: {do_tautomer}")
    
    # -------------------------------------------------------------------------
    # Run standardization
    # -------------------------------------------------------------------------
    if engine == "mp":
        df = standardize_multiprocess(
            df, src_col, do_tautomer,
            max(1, args.workers),
            args.chunksize
        )
    else:
        df = standardize_pandas(df, src_col, do_tautomer)
    
    # Remove invalid entries
    df = df.dropna(subset=["smiles_neutral"])
    print(f"[Standardize] Valid structures: {len(df):,}")
    
    # -------------------------------------------------------------------------
    # Deduplicate
    # -------------------------------------------------------------------------
    if args.dedup:
        df = deduplicate_data(df, args.dedup_thresh)
    
    # -------------------------------------------------------------------------
    # Generate unique IDs
    # -------------------------------------------------------------------------
    df = generate_row_uids(df)
    
    # -------------------------------------------------------------------------
    # Cleanup and save
    # -------------------------------------------------------------------------
    cols_to_drop = ["cond_uid", "temp_C_grp", "solvent_grp", "smiles_original"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df.to_parquet(args.out, index=False)
    print(f"[Standardize] Saved: {args.out}")
    
    if args.export_csv:
        csv_path = os.path.splitext(args.out)[0] + ".csv"
        df.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
        print(f"[Standardize] CSV exported: {csv_path}")
    
    # Final summary
    print(f"[Standardize] Complete. {len(df):,} rows ({initial_rows - len(df):,} removed)")


if __name__ == "__main__":
    main()
