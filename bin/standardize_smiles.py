#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standardize SMILES -> smiles_neutral (+ Full InChIKey).
Includes multiprocessing support for large datasets.
"""

import argparse
import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# RDKit
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

# ---------- Standardizer (Process Cache) ----------
_STD = None

def _make_standardizer():
    """Initializes RDKit standardizer objects."""
    normalizer = rdMolStandardize.Normalizer()
    reionizer  = rdMolStandardize.Reionizer()
    uncharger  = rdMolStandardize.Uncharger()
    largest    = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)

    # RDKit version compatibility
    try:
        taut_can = rdMolStandardize.TautomerCanonicalizer()
        def canonize(m): return taut_can.canonicalize(m)
    except AttributeError:
        te = rdMolStandardize.TautomerEnumerator()
        def canonize(m): return te.Canonicalize(m)

    return normalizer, reionizer, uncharger, largest, canonize

def _ensure_std():
    """Singleton accessor for the standardizer in each worker process."""
    global _STD
    if _STD is None:
        _STD = _make_standardizer()
    return _STD


# ---------- Atomic Functions ----------
def _std_one(smi: str, do_tautomer: bool):
    """
    Standardizes a single SMILES string:
      - Largest fragment -> Normalize -> Reionize -> Uncharge
      - Optional: Tautomer Canonicalization
      - Returns: (smiles_neutral, full_InChIKey)
    """
    if not smi or str(smi).lower() == 'nan':
        return (None, None)
    
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (None, None)
        
    normalizer, reionizer, uncharger, largest, canonize = _ensure_std()
    try:
        mol = largest.choose(mol)
        mol = normalizer.normalize(mol)
        mol = reionizer.reionize(mol)
        mol = uncharger.uncharge(mol)
        
        if do_tautomer:
            mol = canonize(mol)
            
        Chem.SanitizeMol(mol)
        smi_neu = Chem.MolToSmiles(mol, canonical=True)
        ik = inchi.MolToInchiKey(mol)  # Full InChIKey
        return (smi_neu, ik)
    except Exception:
        return (None, None)


# ---------- Executors ----------
def _process_chunk(idx_range, smi_series, do_tautomer):
    """Worker function to process a chunk of SMILES."""
    out_smi, out_ik = [], []
    for s in smi_series:
        smi_neu, ik = _std_one(s, do_tautomer)
        out_smi.append(smi_neu)
        out_ik.append(ik)
    return (idx_range, out_smi, out_ik)

def _fill_inchikey(df: pd.DataFrame, inchikey_vec, overwrite_inchi: bool):
    """
    Fills/Creates 'InChIKey' column based on policy:
      - If missing -> Create
      - If exists + overwrite -> Replace
      - If exists + no overwrite -> Fill NaNs only
    """
    if "InChIKey" not in df.columns:
        df["InChIKey"] = pd.Series(inchikey_vec, dtype="string")
        return df

    if overwrite_inchi:
        df["InChIKey"] = pd.Series(inchikey_vec, dtype="string")
        return df

    # Fill gaps
    s = df["InChIKey"].astype("string")
    mask = s.isna() | s.str.lower().eq("nan") | s.str.len().fillna(0).eq(0)
    df.loc[mask, "InChIKey"] = pd.Series(inchikey_vec, dtype="string")
    return df

def standardize_mp(df, src_col, do_tautomer, workers, chunksize, overwrite_inchi):
    """Multiprocessing execution engine."""
    n = len(df)
    if n == 0:
        df["smiles_neutral"] = pd.Series(dtype="string")
        if "InChIKey" not in df.columns:
            df["InChIKey"] = pd.Series(dtype="string")
        return df

    if chunksize <= 0:
        chunksize = max(1000, n // max(1, (workers * 4)))

    ranges = [(start, min(start + chunksize, n)) for start in range(0, n, chunksize)]

    smiles_neu = [None] * n
    inchikey   = [None] * n

    print(f"[Standardize] Running with {workers} workers on {n} rows...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for (start, stop) in ranges:
            sub = df[src_col].iloc[start:stop].astype("string")
            futs.append(ex.submit(_process_chunk, (start, stop), sub, do_tautomer))

        for fut in as_completed(futs):
            (start, stop), smi_list, ik_list = fut.result()
            smiles_neu[start:stop] = smi_list
            inchikey[start:stop]   = ik_list

    df["smiles_neutral"] = pd.Series(smiles_neu, dtype="string")
    df = _fill_inchikey(df, inchikey, overwrite_inchi=overwrite_inchi)
    return df

def standardize_pandas(df, src_col, do_tautomer, overwrite_inchi):
    """Single-threaded pandas execution engine."""
    print("[Standardize] Running in single-threaded mode...")
    pairs = df[src_col].astype("string").apply(lambda s: _std_one(s, do_tautomer))
    
    df["smiles_neutral"] = pairs.map(lambda t: t[0]).astype("string")
    inchikey_vec = pairs.map(lambda t: t[1]).astype("string")
    
    df = _fill_inchikey(df, inchikey_vec, overwrite_inchi=overwrite_inchi)
    return df


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Standardize SMILES -> smiles_neutral (+ Full InChIKey).")
    ap.add_argument("--in",  dest="inp", required=True, help="Input Parquet file")
    ap.add_argument("--out", dest="out", required=True, help="Output Parquet file")
    ap.add_argument("--overwrite-inchikey", action="store_true", help="Recompute/overwrite InChIKey from neutral SMILES.")
    ap.add_argument("--no-tautomer", action="store_true", help="Disable tautomer canonicalization.")
    ap.add_argument("--engine", choices=["auto","mp","pandas"], default="auto", help="Execution engine.")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Num workers for mp.")
    ap.add_argument("--chunksize", type=int, default=2000, help="Chunk size for mp.")
    ap.add_argument("--export-csv", action="store_true", help="Also export CSV.")
    
    args = ap.parse_args()

    print(f"[Standardize] Loading {args.inp}...")
    try:
        df = pd.read_parquet(args.inp)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read input: {e}")

    # Determine source column
    src_col = None
    if "smiles_neutral" in df.columns and df["smiles_neutral"].notna().any():
        src_col = "smiles_neutral"
    elif "smiles_original" in df.columns:
        src_col = "smiles_original"
    
    if src_col is None:
        sys.exit("[ERROR] No 'smiles_original' or 'smiles_neutral' column found.")

    do_taut = not args.no_tautomer

    # Select Engine
    n = len(df)
    engine = args.engine
    if engine == "auto":
        engine = "mp" if n >= 5000 else "pandas"

    if engine == "mp":
        df = standardize_mp(df, src_col, do_taut, max(1, args.workers), args.chunksize, args.overwrite_inchikey)
    else:
        df = standardize_pandas(df, src_col, do_taut, args.overwrite_inchikey)

    # Cast columns to string type for consistency
    for c in ["source","smiles_original","smiles_neutral","InChIKey","solvent",
              "target_unit_raw","target_family","strat_label","method"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # === Generate UIDs (Compound | Condition | Row) ===
    def _norm_str(s):
        return (s.astype("string").fillna("").str.strip())

    def _norm_temp(s):
        return pd.to_numeric(s, errors="coerce")

    if "InChIKey" not in df.columns: df["InChIKey"] = pd.NA
    if "solvent" not in df.columns:  df["solvent"]  = pd.NA
    if "temp_C" not in df.columns:   df["temp_C"]   = pd.NA

    ik   = _norm_str(df["InChIKey"])
    solv = _norm_str(df["solvent"])
    temp = _norm_temp(df["temp_C"])

    # cond_uid = Compound + Solvent + Temp
    cond_uid = (
        ik.fillna("") + "|" +
        solv.fillna("") + "|" +
        temp.fillna(pd.NA).astype("string").fillna("")
    )
    df["cond_uid"] = cond_uid.astype("string")

    # row_uid = cond_uid + replica_index
    rep_idx = df.groupby("cond_uid").cumcount()
    df["row_uid"] = (df["cond_uid"] + "#" + rep_idx.astype(str)).astype("string")

    # Save
    out_parquet = args.out
    df.to_parquet(out_parquet, index=False)

    if args.export_csv:
        csv_path = os.path.splitext(out_parquet)[0] + ".csv"
        df.to_csv(csv_path, index=False, lineterminator="\n", encoding="utf-8")

    print(f"[Standardize] Success: {len(df)} rows -> {out_parquet}")

if __name__ == "__main__":
    main()