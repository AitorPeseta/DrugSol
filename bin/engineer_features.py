#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engineer_features.py
--------------------
Calculates physical and chemical features for the dataset:
1. Gaussian Sample Weights (prioritizing 37°C).
2. Quantitative Estimation of Drug-likeness (QED).
3. Ionization counts (Acid/Base) using SMARTS.
4. pKa prediction via MolGpKa API (With imputation for neutrals).
"""

import argparse
import math
import json
import sys
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # Progress bar

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED

# ============================================================================
# 1. WEIGHTS (Gaussian Distribution)
# ============================================================================

def calculate_gaussian_weights(df, temp_col='temp_C', target_temp=37.0, sigma=8.0):
    print(f"[Features] Calculating Gaussian weights centered at {target_temp}°C...")
    t = pd.to_numeric(df[temp_col], errors='coerce')
    base_weight = 1.0
    bonus_weight = 2.0
    exp_term = np.exp(-((t - target_temp) ** 2) / (2 * sigma ** 2))
    weights = base_weight + (bonus_weight * exp_term)
    return weights.fillna(1.0)

# ============================================================================
# 2. DRUG-LIKENESS (QED)
# ============================================================================

def calculate_qed_features(df, smiles_col):
    print("[Features] Calculating QED (Drug-likeness)...")
    def _get_qed(smi):
        if not smi: return 0.0
        mol = Chem.MolFromSmiles(smi)
        if not mol: return 0.0
        try:
            return QED.qed(mol)
        except:
            return 0.0
    return df[smiles_col].apply(_get_qed)

# ============================================================================
# 3. IONIZATION (SMARTS)
# ============================================================================

def split_acid_base_pattern(smarts_file):
    try:
        df_smarts = pd.read_csv(smarts_file, sep="\t")
    except FileNotFoundError:
        sys.exit(f"[ERROR] SMARTS file not found: {smarts_file}")
    df_acid = df_smarts[df_smarts.Acid_or_base == "A"].reset_index(drop=True)
    df_base = df_smarts[df_smarts.Acid_or_base == "B"].reset_index(drop=True)
    return df_acid, df_base

def unique_acid_match(matches):
    single_matches = list(set([m[0] for m in matches if len(m) == 1]))
    double_matches = [m for m in matches if len(m) == 2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches

def match_pattern(df_smarts, mol):
    matches = []
    for row in df_smarts.itertuples():
        try:
            smarts_str = row.SMARTS
            index_str = row.new_index
        except AttributeError:
            continue 
        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None: continue
        match = mol.GetSubstructMatches(pattern)
        if not match: continue
        if isinstance(index_str, str) and "," in index_str:
            idxs = [int(i) for i in index_str.split(",")]
            for m in match:
                matches.append([m[idxs[0]], m[idxs[1]]])
        else:
            try:
                idx = int(index_str)
            except: continue
            for m in match:
                if idx < len(m): matches.append([m[idx]])
    matches = unique_acid_match(matches)
    flat_matches = [item for sublist in matches for item in sublist]
    return list(set(flat_matches))

def calculate_ionization(smiles, df_acid, df_base):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series({"n_ionizable": 0, "n_acid": 0, "n_base": 0})
    mol = AllChem.AddHs(mol)
    acid_idx = match_pattern(df_acid, mol)
    base_idx = match_pattern(df_base, mol)
    return pd.Series({
        "n_ionizable": int(len(acid_idx) + len(base_idx)),
        "n_acid": int(len(acid_idx)),
        "n_base": int(len(base_idx))
    })

# ============================================================================
# 4. PHENOLS
# ============================================================================

_PHENOL_STRICT = Chem.MolFromSmarts("c[OX2H]")
def calculate_phenols(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series({"n_phenol": 0, "has_phenol": 0})
    matches = mol.GetSubstructMatches(_PHENOL_STRICT)
    n = len({m[1] for m in matches if len(m) > 1}) if matches else 0
    return pd.Series({"n_phenol": int(n), "has_phenol": int(n > 0)})

# ============================================================================
# 5. pKa (MolGpKa API)
# ============================================================================

def predict_pka_api(smiles, api_url, token):
    if not smiles: return None
    try:
        resp = requests.post(
            url=api_url, 
            files={"Smiles": ("tmg", smiles)}, 
            headers={"token": token}, 
            timeout=10
        )
        if resp.status_code != 200: return None
        res_json = resp.json()
        if res_json.get("status") != 200: return None
        return res_json.get("gen_datas")
    except:
        return None

def summarize_pka(val):
    default = {
        "pka_acid_min": np.nan, "pka_acid_max": np.nan, 
        "pka_base_min": np.nan, "pka_base_max": np.nan
    }
    if not val or not isinstance(val, dict): return pd.Series(default)
    
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
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--temp-col", default="temp_C")
    ap.add_argument("--smarts", required=True)
    ap.add_argument("--pka-api-url", default="http://xundrug.cn:5001/modules/upload0/")
    ap.add_argument("--pka-token", default=None)
    
    args = ap.parse_args()

    print(f"[Features] Loading {args.inp}...")
    if args.inp.endswith(".parquet"):
        df = pd.read_parquet(args.inp)
    else:
        df = pd.read_csv(args.inp)

    if args.smiles_col not in df.columns:
        sys.exit(f"[ERROR] SMILES column '{args.smiles_col}' not found.")

    # 2. Gaussian Weights
    df["weight"] = calculate_gaussian_weights(df, args.temp_col)

    # 3. QED
    df["qed"] = calculate_qed_features(df, args.smiles_col)

    # 4. Ionization
    print("[Features] Calculating Ionization (SMARTS)...")
    df_acid, df_base = split_acid_base_pattern(args.smarts)
    tqdm.pandas(desc="Ionization")
    ion_feats = df[args.smiles_col].progress_apply(lambda x: calculate_ionization(x, df_acid, df_base))
    df = pd.concat([df, ion_feats], axis=1)

    # 5. Phenols
    print("[Features] Calculating Phenols...")
    phenol_feats = df[args.smiles_col].progress_apply(calculate_phenols)
    df = pd.concat([df, phenol_feats], axis=1)

    # 6. pKa API
    if args.pka_token:
        print("[Features] Querying pKa API...")
        cache = {} 
        pka_results = []
        for smi in tqdm(df[args.smiles_col], desc="pKa API"):
            smi_key = str(smi)
            if smi_key not in cache:
                res = predict_pka_api(smi_key, args.pka_api_url, args.pka_token)
                cache[smi_key] = summarize_pka(res)
            pka_results.append(cache[smi_key])
            
        pka_df = pd.DataFrame(pka_results)
        
        # --- IMPUTATION FIX START ---
        # Fill NaNs with physically meaningful "extreme" values
        # Acid NaN -> 16.0 (Very weak acid, effectively neutral)
        pka_df["pka_acid_min"] = pka_df["pka_acid_min"].fillna(16.0)
        pka_df["pka_acid_max"] = pka_df["pka_acid_max"].fillna(16.0)
        
        # Base NaN -> -2.0 (Very weak base, effectively neutral)
        pka_df["pka_base_min"] = pka_df["pka_base_min"].fillna(-2.0)
        pka_df["pka_base_max"] = pka_df["pka_base_max"].fillna(-2.0)
        # --- IMPUTATION FIX END ---

        df = pd.concat([df.reset_index(drop=True), pka_df.reset_index(drop=True)], axis=1)
    else:
        print("[INFO] Skipping pKa calculation.")

    print(f"[Features] Saving to {args.out}...")
    df.to_parquet(args.out, index=False)
    print("[Features] Done.")

if __name__ == "__main__":
    main()