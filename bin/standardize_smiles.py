#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standardize_smiles.py (Fix: Pandas NA handling)
"""

import argparse, os, sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# RDKit
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

# ---------- Standardizer (cache por proceso) ----------
_STD = None
def _make_standardizer():
    normalizer = rdMolStandardize.Normalizer()
    reionizer  = rdMolStandardize.Reionizer()
    uncharger  = rdMolStandardize.Uncharger()
    
    try:
        taut_can = rdMolStandardize.TautomerCanonicalizer()
        def canonize(m): return taut_can.canonicalize(m)
    except AttributeError:
        te = rdMolStandardize.TautomerEnumerator()
        def canonize(m): return te.Canonicalize(m)
    return normalizer, reionizer, uncharger, canonize

def _ensure_std():
    global _STD
    if _STD is None:
        _STD = _make_standardizer()
    return _STD

def _std_one(smi, do_tautomer: bool):
    """
    Devuelve (SMILES_CANONICO, INCHIKEY) o (None, None).
    Maneja robustamente pd.NA y nulos.
    """
    # 1. Chequeo robusto de nulos (Fix para TypeError: boolean value of NA is ambiguous)
    if pd.isna(smi):
        return (None, None)

    # 2. Convertir a string seguro
    s_smi = str(smi).strip()
    if not s_smi or s_smi.lower() == 'nan':
        return (None, None)
    
    # 3. Filtro Rápido de Sales (String)
    if "." in s_smi:
        return (None, None)

    # 4. RDKit Parsing
    mol = Chem.MolFromSmiles(s_smi)
    if mol is None:
        return (None, None)

    # 5. Filtro Robusto de Sales (Fragmentos)
    if len(Chem.GetMolFrags(mol)) > 1:
        return (None, None)

    normalizer, reionizer, uncharger, canonize = _ensure_std()
    try:
        mol = normalizer.normalize(mol)
        mol = reionizer.reionize(mol)
        mol = uncharger.uncharge(mol)
        
        if do_tautomer:
            mol = canonize(mol)
            
        Chem.SanitizeMol(mol)
        
        smi_neu = Chem.MolToSmiles(mol, canonical=True)
        ik = inchi.MolToInchiKey(mol)
        return (smi_neu, ik)
    except Exception:
        return (None, None)

# ---------- Ejecutores ----------
def _process_chunk(idx_range, smi_series, do_tautomer):
    out_smi, out_ik = [], []
    for s in smi_series:
        smi_neu, ik = _std_one(s, do_tautomer)
        out_smi.append(smi_neu)
        out_ik.append(ik)
    return (idx_range, out_smi, out_ik)

def standardize_mp(df, src_col, do_tautomer, workers, chunksize):
    n = len(df)
    if n == 0:
        df["smiles_neutral"] = pd.Series(dtype="string")
        df["InChIKey"] = pd.Series(dtype="string")
        return df

    if chunksize <= 0:
        chunksize = max(1000, n // max(1, (workers * 4)))

    ranges = [(start, min(start + chunksize, n)) for start in range(0, n, chunksize)]
    smiles_neu = [None] * n
    inchikey   = [None] * n

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for (start, stop) in ranges:
            # Pasamos como lista o array para evitar líos de índices en MP
            sub = df[src_col].iloc[start:stop].tolist()
            futs.append(ex.submit(_process_chunk, (start, stop), sub, do_tautomer))

        for fut in as_completed(futs):
            (start, stop), smi_list, ik_list = fut.result()
            smiles_neu[start:stop] = smi_list
            inchikey[start:stop]   = ik_list

    df["smiles_neutral"] = pd.Series(smiles_neu, dtype="string")
    df["InChIKey"] = pd.Series(inchikey, dtype="string")
    return df

def standardize_pandas(df, src_col, do_tautomer):
    # Usamos map con la función corregida
    # Convertimos a object/list antes para evitar el comportamiento raro de .apply en StringDtype
    pairs = df[src_col].astype(object).apply(lambda s: _std_one(s, do_tautomer))
    
    df["smiles_neutral"] = pairs.map(lambda t: t[0]).astype("string")
    df["InChIKey"] = pairs.map(lambda t: t[1]).astype("string")
    return df

# ---------- DEDUPLICACIÓN INTELIGENTE ----------
def deduplicate_data(df, threshold=0.7):
    print(f"[Dedup] Iniciando deduplicación por InChIKey (thresh={threshold})...")
    
    # Importante: dropna aquí también por si acaso
    df = df.dropna(subset=["InChIKey"])
    
    df["temp_C_grp"] = df["temp_C"].fillna(-9999)
    df["solvent_grp"] = df["solvent"].fillna("UNKNOWN")
    
    def _agg_logic(x):
        vals = x.values
        if len(vals) == 1:
            return vals[0]
        
        med = np.median(vals)
        mask = np.abs(vals - med) <= threshold
        clean_vals = vals[mask]
        
        if len(clean_vals) == 0:
            return np.nan 
        
        return np.mean(clean_vals)

    dup_mask = df.duplicated(subset=["InChIKey", "solvent_grp", "temp_C_grp"], keep=False)
    
    df_unique = df[~dup_mask].copy()
    df_dups = df[dup_mask].copy()
    
    if df_dups.empty:
        print("[Dedup] No se encontraron duplicados.")
        return df.drop(columns=["temp_C_grp", "solvent_grp"])
    
    print(f"[Dedup] Procesando {len(df_dups)} filas duplicadas...")
    
    grp = df_dups.groupby(["InChIKey", "solvent_grp", "temp_C_grp"])
    new_logs = grp["logS"].apply(_agg_logic)
    
    df_collapsed = grp.first().reset_index()
    
    # Alinear índices de logS calculado
    # Al hacer reset_index, el orden debería coincidir con new_logs.values si el sort es consistente
    # Para máxima seguridad, hacemos un map usando el índice múltiple, pero values suele funcionar en groupby default
    df_collapsed["logS"] = new_logs.values
    
    df_collapsed = df_collapsed.dropna(subset=["logS"])
    
    final_cols = [c for c in df.columns if c not in ["temp_C_grp", "solvent_grp"]]
    df_clean = pd.concat([df_unique[final_cols], df_collapsed[final_cols]], ignore_index=True)
    
    print(f"[Dedup] Final: {len(df)} -> {len(df_clean)} filas únicas.")
    return df_clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--overwrite-inchikey", action="store_true")
    ap.add_argument("--no-tautomer", action="store_true")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--export-csv", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--dedup-thresh", type=float, default=0.7)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)

    src_col = None
    if "smiles_neutral" in df.columns and df["smiles_neutral"].notna().any():
        src_col = "smiles_neutral"
    elif "smiles_original" in df.columns:
        src_col = "smiles_original"
    
    if src_col is None:
        sys.exit("[ERROR] No smiles column found.")

    do_taut = not args.no_tautomer
    engine = args.engine
    if engine == "auto": engine = "mp" if len(df) >= 5000 else "pandas"

    print(f"[Std] Estandarizando desde '{src_col}'...")
    
    if engine == "mp":
        df = standardize_mp(df, src_col, do_taut, max(1, args.workers), args.chunksize)
    else:
        df = standardize_pandas(df, src_col, do_taut)

    # Eliminar inválidos
    df = df.dropna(subset=["smiles_neutral"])

    # Deduplicar
    if args.dedup:
        df = deduplicate_data(df, args.dedup_thresh)

    # UIDs
    def _norm_str(s): return (s.astype("string").fillna("UNKNOWN").str.strip())
    def _norm_temp(s): return pd.to_numeric(s, errors="coerce")

    ik   = _norm_str(df["InChIKey"])
    solv = _norm_str(df["solvent"])
    temp = _norm_temp(df["temp_C"])
    temp_str = temp.apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NA")

    df["row_uid"] = (ik + "|" + solv + "|" + temp_str).astype("string")

    if df["row_uid"].duplicated().any():
        n_dup = df["row_uid"].duplicated().sum()
        print(f"[WARN] {n_dup} UIDs duplicados. Forzando eliminación.")
        df = df.drop_duplicates(subset=["row_uid"], keep="first")

    # Limpieza
    cols_drop = ["cond_uid", "temp_C_grp", "solvent_grp", "smiles_original"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])

    df.to_parquet(args.out, index=False)
    if args.export_csv:
        df.to_csv(os.path.splitext(args.out)[0] + ".csv", index=False)

    print(f"[Standardize] Hecho. {len(df)} filas.")

if __name__ == "__main__":
    main()