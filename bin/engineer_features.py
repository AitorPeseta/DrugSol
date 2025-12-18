#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engineer_features.py
--------------------
Calculates physical and chemical features for the dataset:
1. Balanced Sample Weights (Temperature + Solubility Rarity).
2. Ionization counts (Acid/Base) using SMARTS.
3. pKa prediction via MolGpKa API (With imputation for neutrals).
4. Phenols count.
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed # <--- NUEVO

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# ----------------------------------------------------------------------------
# WRAPPERS PARA PARALELIZACIÓN (NUEVO)
# ----------------------------------------------------------------------------

def _process_chunk_ionization(chunk_tuple):
    """Función auxiliar para procesar un trozo del dataframe en paralelo"""
    # Desempaquetamos argumentos
    idx, series, df_acid, df_base = chunk_tuple
    # Aplicamos la lógica fila por fila en este trozo
    res = series.apply(lambda x: calculate_ionization(x, df_acid, df_base))
    return idx, res

def _process_chunk_phenols(chunk_tuple):
    idx, series = chunk_tuple
    res = series.apply(calculate_phenols)
    return idx, res

# ============================================================================
# 1. WEIGHTS (Balanced: Temperature + Solubility Rarity)
# ============================================================================

def calculate_balanced_weights(df, temp_col='temp_C', target_col='logS', target_temp=37.0, sigma=5.0):
    """
    Calcula pesos priorizando agresivamente la temperatura fisiológica (37°C) 
    sobre la temperatura ambiente (25°C), manteniendo un balance suave por clases.
    
    Cambios aplicados para corregir sesgo de temperatura:
    1. Sigma reducido (8.0 -> 5.0): La campana de Gauss es más estrecha. 25°C ya no se considera "cerca".
    2. Bonus aumentado (2.0 -> 4.0): Los datos a 37°C valen 5x más que la base.
    3. Fórmula final cuadrática: w_final = w_class * (w_temp ** 2).
    """
    print(f"[Features] Calculating Balanced Weights (Target Temp={target_temp}°C + LogS Rarity)...")
    
    # Peso por Temperatura (Gaussiana Agresiva)
    # Rellenamos NaNs con 25.0 (asumimos ambiente si no hay dato)
    t = pd.to_numeric(df[temp_col], errors='coerce').fillna(25.0)
    
    base_weight = 1.0
    bonus_weight = 4.0  # Aumentado para dar mucha más importancia al pico
    
    # Cálculo Gaussiano
    # A 37°C: exp(0) = 1 -> w = 1 + 4 = 5
    # A 25°C: exp(-2.88) ≈ 0.05 -> w = 1 + 0.2 = 1.2
    exp_term = np.exp(-((t - target_temp) ** 2) / (2 * sigma ** 2))
    w_temp = base_weight + (bonus_weight * exp_term)
    
    # Peso por Rareza de Solubilidad (Suavizado)
    # Seguimos usándolo para que no ignore totalmente las clases minoritarias,
    # pero confiamos en que w_temp hará el trabajo pesado.
    if target_col in df.columns:
        logs = pd.to_numeric(df[target_col], errors='coerce').fillna(-2.0)
        
        # Bins desde -12 a 4
        bins = np.linspace(-12, 4, 17) 
        counts, _ = np.histogram(logs, bins=bins)
        
        counts = np.maximum(counts, 1) # Evitar div/0
        total_samples = len(df)
        
        indices = np.digitize(logs, bins) - 1
        indices = np.clip(indices, 0, len(counts)-1)
        row_counts = counts[indices]
        
        # Log(1 + Total/Count) suele dar valores entre 1 y 5.
        w_solubility = np.log1p(total_samples / row_counts)
        
        print(f"   -> Solubility Weights range: {w_solubility.min():.2f} - {w_solubility.max():.2f}")
    else:
        print(f"[WARN] Target column '{target_col}' not found. Using only Temperature weights.")
        w_solubility = 1.0

    # C. Combinación CUADRÁTICA
    # Elevamos la temperatura al cuadrado para castigar severamente el sesgo de 25°C
    # Ejemplo Ratio: 
    # 37°C -> 5^2 = 25
    # 25°C -> 1.2^2 = 1.44
    # Un dato a 37°C vale ahora ~17 veces más que uno a 25°C.
    final_weight = w_solubility * (w_temp ** 2)
    
    print(f"   -> Final Weights (Mean): {final_weight.mean():.2f}")
    return final_weight.fillna(1.0)

# ============================================================================
# 2. IONIZATION (SMARTS)
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
    # (Misma lógica que tenías antes)
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
# 3. PHENOLS
# ============================================================================

_PHENOL_STRICT = Chem.MolFromSmarts("c[OX2H]")
def calculate_phenols(smiles):
    # (Misma lógica que tenías antes, redefinimos _PHENOL_STRICT dentro o global)
    _PHENOL_STRICT = Chem.MolFromSmarts("c[OX2H]") 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series({"n_phenol": 0, "has_phenol": 0})
    matches = mol.GetSubstructMatches(_PHENOL_STRICT)
    n = len({m[1] for m in matches if len(m) > 1}) if matches else 0
    return pd.Series({"n_phenol": int(n), "has_phenol": int(n > 0)})

# ============================================================================
# 4. pKa (MolGpKa API)
# ============================================================================

def predict_pka_api(smiles, api_url, token):
    # (Misma lógica que tenías antes)
    if not smiles: return None
    try:
        resp = requests.post(
            url=api_url, 
            files={"Smiles": ("tmg", smiles)}, 
            headers={"token": token}, 
            timeout=20 # Subimos un poco el timeout por si acaso
        )
        if resp.status_code != 200: return None
        res_json = resp.json()
        if res_json.get("status") != 200: return None
        return res_json.get("gen_datas")
    except:
        return None
    
def summarize_pka(val):
    # (Misma lógica que tenías antes)
    default = {
        "pka_acid_min": np.nan, "pka_acid_max": np.nan, 
        "pka_base_min": np.nan, "pka_base_max": np.nan
    }
    if not val or not isinstance(val, dict): return pd.Series(default)
    acid = val.get("Acid", {})
    base = val.get("Base", {})
    def get_vals(d): return [float(v) for v in d.values() if v]
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
    ap.add_argument("--nproc", type=int, default=1, help="Number of CPUs")

    args = ap.parse_args()

    # Ajuste de CPUs: Si es 1, usa pandas normal. Si >1, usa multiprocessing.
    n_jobs = max(1, args.nproc)
    print(f"[Features] Running with {n_jobs} CPUs")

    print(f"[Features] Loading {args.inp}...")
    if args.inp.endswith(".parquet"):
        df = pd.read_parquet(args.inp)
    else:
        df = pd.read_csv(args.inp)

    if args.smiles_col not in df.columns:
        sys.exit(f"[ERROR] SMILES column '{args.smiles_col}' not found.")

    # 1. Balanced Weights (Vectorizado, ya es rápido, usa 1 CPU)
    from engineer_features import calculate_balanced_weights # O define la función arriba
    # Nota: Si copiaste la función arriba, llama directamente a calculate_balanced_weights(df...)
    df["weight"] = calculate_balanced_weights(df, args.temp_col, target_col='logS')

    # Preparar datos SMARTS una sola vez
    df_acid, df_base = split_acid_base_pattern(args.smarts)

    print("[Features] Calculating Ionization (Parallel)...")
    if n_jobs > 1:
        # Dividimos el DF en chunks
        chunks = np.array_split(df[args.smiles_col], n_jobs * 4) # *4 para mejor balanceo
        
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Enviamos tareas
            futures = [executor.submit(_process_chunk_ionization, (i, chunk, df_acid, df_base)) 
                       for i, chunk in enumerate(chunks)]
            
            # Recogemos resultados con barra de progreso
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Ionization MP"):
                results.append(future.result())
        
        # Reordenamos y concatenamos
        results.sort(key=lambda x: x[0])
        ion_feats = pd.concat([res[1] for res in results])
    else:
        tqdm.pandas(desc="Ionization")
        ion_feats = df[args.smiles_col].progress_apply(lambda x: calculate_ionization(x, df_acid, df_base))

    df = pd.concat([df, ion_feats], axis=1)

    print("[Features] Calculating Phenols (Parallel)...")
    if n_jobs > 1:
        chunks = np.array_split(df[args.smiles_col], n_jobs * 4)
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_process_chunk_phenols, (i, chunk)) for i, chunk in enumerate(chunks)]
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Phenols MP"):
                results.append(future.result())
        results.sort(key=lambda x: x[0])
        phenol_feats = pd.concat([res[1] for res in results])
    else:
        phenol_feats = df[args.smiles_col].progress_apply(calculate_phenols)
        
    df = pd.concat([df, phenol_feats], axis=1)

    
    if args.pka_token:
        print("[Features] Querying pKa API...")
        unique_smiles = df[args.smiles_col].unique()
        cache = {}
        
        # Usamos ThreadPoolExecutor porque es I/O (Red), no gasta CPU.
        # Ponemos max_workers=8 para no saturar al servidor de la API.
        api_threads = min(8, n_jobs * 2) 
        
        with ThreadPoolExecutor(max_workers=api_threads) as executor:
            # Diccionario futuro -> smiles
            future_to_smi = {
                executor.submit(predict_pka_api, str(smi), args.pka_api_url, args.pka_token): str(smi) 
                for smi in unique_smiles
            }
            
            for future in tqdm(as_completed(future_to_smi), total=len(unique_smiles), desc="pKa API Threads"):
                smi = future_to_smi[future]
                try:
                    res = future.result()
                    cache[smi] = summarize_pka(res)
                except Exception as e:
                    cache[smi] = summarize_pka(None) # Fallback

        # Mapeamos los resultados al DF original
        pka_results = df[args.smiles_col].astype(str).map(cache).apply(pd.Series)
        
        # Imputation
        pka_results["pka_acid_min"] = pka_results["pka_acid_min"].fillna(16.0)
        pka_results["pka_acid_max"] = pka_results["pka_acid_max"].fillna(16.0)
        pka_results["pka_base_min"] = pka_results["pka_base_min"].fillna(-2.0)
        pka_results["pka_base_max"] = pka_results["pka_base_max"].fillna(-2.0)

        df = pd.concat([df, pka_results], axis=1)
    else:
        print("[INFO] Skipping pKa calculation.")

    print(f"[Features] Saving to {args.out}...")
    df.to_parquet(args.out, index=False)
    print("[Features] Done.")

if __name__ == "__main__":
    main()