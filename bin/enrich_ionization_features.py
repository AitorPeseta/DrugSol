#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# RDKit (solo para SMILES y SMARTS matching; no descriptores)
from rdkit import Chem
from rdkit.Chem import AllChem
import requests


# ============== SMARTS helpers (tu código adaptado) ==============

def split_acid_base_pattern(smarts_file: str):
    df_smarts = pd.read_csv(smarts_file, sep="\t")
    df_smarts_acid = df_smarts[df_smarts.Acid_or_base == "A"]
    df_smarts_base = df_smarts[df_smarts.Acid_or_base == "B"]
    return df_smarts_acid, df_smarts_base

def unique_acid_match(matches):
    single_matches = list(set([m[0] for m in matches if len(m) == 1]))
    double_matches = [m for m in matches if len(m) == 2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches

def match_acid(df_smarts_acid, mol):
    matches = []
    for _, name, smarts, index, _, acid_base in df_smarts_acid.itertuples():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            found = mol.GetSubstructMatches(pattern)
            if len(found) == 0:
                continue
            if len(index) > 2:
                idxs = [int(i) for i in index.split(",")]
                for m in found:
                    matches.append([m[idxs[0]], m[idxs[1]]])
            else:
                ii = int(index)
                for m in found:
                    matches.append([m[ii]])
        except Exception:
            continue
    matches = unique_acid_match(matches)
    flat = []
    for i in matches:
        for j in i:
            flat.append(j)
    return list(set(flat))

def match_base(df_smarts_base, mol):
    matches = []
    for _, name, smarts, index, _, acid_base in df_smarts_base.itertuples():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            found = mol.GetSubstructMatches(pattern)
            if len(found) == 0:
                continue
            if len(index) > 2:
                idxs = [int(i) for i in index.split(",")]
                for m in found:
                    matches.append([m[idxs[0]], m[idxs[1]]])
            else:
                ii = int(index)
                for m in found:
                    matches.append([m[ii]])
        except Exception:
            continue
    matches = unique_acid_match(matches)
    flat = []
    for i in matches:
        for j in i:
            flat.append(j)
    return list(set(flat))

def get_ionization_aid(mol, smarts_file: str):
    try:
        df_smarts_acid, df_smarts_base = split_acid_base_pattern(smarts_file)
        acid_idx = match_acid(df_smarts_acid, mol)
        base_idx = match_base(df_smarts_base, mol)
        return acid_idx, base_idx
    except Exception:
        return [], []


# ============== pKa API opcional ==============

def predict_pka_api(smi: str, url: str, token: str):
    param = {"Smiles": ("tmg", smi)}
    headers = {"token": token} if token else {}
    r = requests.post(url=url, files=param, headers=headers)
    jsonbool = int(r.headers.get('ifjson', 0))
    if jsonbool == 1:
        res_json = r.json()
        if res_json.get('status') == 200:
            pka_datas = res_json.get('gen_datas', {})
            return pka_datas
        else:
            raise RuntimeError("Error for pKa prediction")
    else:
        raise RuntimeError("Error for pKa prediction")

def pka_stats_from_maps(acid_map: dict, base_map: dict):
    def to_floats(d):
        arr = []
        for _, v in (d or {}).items():
            try:
                arr.append(float(v))
            except Exception:
                continue
        return arr

    acids = to_floats(acid_map)
    bases = to_floats(base_map)

    def stats(xs):
        if not xs:
            return np.nan, np.nan, np.nan
        xs = sorted(xs)
        return xs[0], float(np.median(xs)), xs[-1]

    pKa_min_acid, pKa_med_acid, pKa_max_acid = stats(acids)
    pKa_min_base, pKa_med_base, pKa_max_base = stats(bases)

    # flags
    has_strong_acid = int(bool(acids) and (min(acids) < 3.0))
    has_strong_base = int(bool(bases) and (max(bases) > 9.0))
    has_near_neutral = int(
        any(6.0 <= x <= 8.0 for x in acids) or any(6.0 <= x <= 8.0 for x in bases)
    )

    # NUEVO: recuento de sitios fuertes por tipo
    num_strong_acid = int(sum(x < 3.0 for x in acids))
    num_strong_basic = int(sum(x > 9.0 for x in bases))

    pKa_gap = np.nan
    if acids and bases:
        pKa_gap = float(max(bases) - min(acids))

    return {
        "pKa_min_acid": pKa_min_acid,
        "pKa_med_acid": pKa_med_acid,
        "pKa_max_acid": pKa_max_acid,
        "pKa_min_base": pKa_min_base,
        "pKa_med_base": pKa_med_base,
        "pKa_max_base": pKa_max_base,
        "pKa_gap": pKa_gap,
        "has_strong_acid": has_strong_acid,
        "has_strong_base": has_strong_base,
        "has_near_neutral": has_near_neutral,
        "num_strong_acid": num_strong_acid,
        "num_strong_basic": num_strong_basic,
    }


# ============== core helpers ==============

def safe_mol_from_smiles(smi: str):
    try:
        if smi is None or (isinstance(smi, float) and math.isnan(smi)):
            return None
        smi = str(smi).strip()
        if not smi:
            return None
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None

# === FENOLES (nuevo) =========================================================
_PHENOL_SMARTS_STRICT = Chem.MolFromSmarts("c[OX2H]")  # O-H en carbono aromático (bencénico)
_PHENOL_SMARTS_LOOSE  = Chem.MolFromSmarts("a[OX2H]")  # O-H en átomo aromático genérico (incluye heteroaromáticos)

def phenol_features(mol: Chem.Mol):
    """
    Devuelve:
      - n_phenol / has_phenol usando patrón 'c[OX2H]' (estricto)
      - n_phenol_like / phenol_like usando 'a[OX2H]' (laxo)
    El conteo se hace por átomos de oxígeno únicos.
    """
    if mol is None:
        return dict(n_phenol=0, has_phenol=0, n_phenol_like=0, phenol_like=0)

    def count_unique_O(smarts):
        if smarts is None:
            return 0
        matches = mol.GetSubstructMatches(smarts)  # tuplas de índices
        if not matches:
            return 0
        # localizar el índice del átomo O en el patrón; en ambos patrones está en la segunda posición
        # (c/a)[OX2H] -> índice 1 del match
        o_idxs = {m[1] for m in matches if len(m) > 1}
        return len(o_idxs)

    n_strict = count_unique_O(_PHENOL_SMARTS_STRICT)
    n_loose  = count_unique_O(_PHENOL_SMARTS_LOOSE)

    return dict(
        n_phenol=int(n_strict),
        has_phenol=int(n_strict > 0),
        n_phenol_like=int(n_loose),
        phenol_like=int(n_loose > 0),
    )
# =============================================================================

def build_features_for_smiles(smi: str, smarts_file: str, pka_url: str = None, pka_token: str = None):
    mol = safe_mol_from_smiles(smi)

    # sitios ácidos/básicos por SMARTS
    if mol is not None:
        try:
            molH = AllChem.AddHs(mol)
        except Exception:
            molH = mol
        acid_idx, base_idx = get_ionization_aid(molH, smarts_file=smarts_file)
    else:
        acid_idx, base_idx = [], []

    n_acid = int(len(acid_idx))
    n_base = int(len(base_idx))
    n_ionizable = int(n_acid + n_base)
    acid_base_balance = int(n_acid - n_base)

    # pKa por tipo (si se ha configurado la API)
    acid_map, base_map = {}, {}
    if pka_url and (n_ionizable > 0) and (mol is not None):
        try:
            out = predict_pka_api(Chem.MolToSmiles(mol), url=pka_url, token=pka_token)
            if "Acid" in out or "Base" in out:
                acid_map = out.get("Acid", {}) or {}
                base_map = out.get("Base", {}) or {}
            elif "Acidic" in out or "Basic" in out:
                acid_map = out.get("Acidic", {}) or {}
                base_map = out.get("Basic", {}) or {}
            else:
                gd = out.get("gen_datas", {}) if isinstance(out, dict) else {}
                acid_map = (gd or {}).get("Acid", {}) or {}
                base_map = (gd or {}).get("Base", {}) or {}
        except Exception:
            acid_map, base_map = {}, {}

    pka_stats = pka_stats_from_maps(acid_map, base_map)

    # === Añadir fenoles ===
    phen = phenol_features(mol)

    out = {
        "n_ionizable": n_ionizable,
        "n_acid": n_acid,
        "n_base": n_base,
        "acid_base_balance": acid_base_balance,
    }
    out.update(pka_stats)
    out.update(phen)
    return out


# ============== I/O ==============

def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(p)
    else:
        raise SystemExit(f"Formato no soportado: {p.suffix}")

def write_like_input(df: pd.DataFrame, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    elif p.suffix.lower() in (".csv", ".txt"):
        df.to_csv(p, index=False)
    else:
        raise SystemExit(f"Formato de salida no soportado: {p.suffix}")

def main():
    ap = argparse.ArgumentParser("Añade SOLO features de ionización+pKa + fenoles (numéricos) al dataset")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--smarts", required=True)
    ap.add_argument("--pka-api-url", default=None)
    ap.add_argument("--pka-token", default=None)
    args = ap.parse_args()

    df = read_any(args.inp).copy()
    if args.smiles_col not in df.columns:
        raise SystemExit(f"Falta columna {args.smiles_col}")

    # temp_C desde temperature_K (útil para tu pipeline, no depende de RDKit)
    if "temperature_K" in df.columns and "temp_C" not in df.columns:
        try:
            df["temp_C"] = df["temperature_K"].astype(float) - 273.15
        except Exception:
            df["temp_C"] = np.nan

    new_cols = [
        "n_ionizable","n_acid","n_base","acid_base_balance",
        "pKa_min_acid","pKa_med_acid","pKa_max_acid",
        "pKa_min_base","pKa_med_base","pKa_max_base",
        "pKa_gap","has_strong_acid","has_strong_base","has_near_neutral",
        "num_strong_acid","num_strong_basic",
        "n_phenol","has_phenol","n_phenol_like","phenol_like",
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = np.nan

    for idx, smi in enumerate(df[args.smiles_col].values):
        try:
            feats = build_features_for_smiles(
                smi,
                smarts_file=args.smarts,
                pka_url=args.pka_api_url,
                pka_token=args.pka_token,
            )
            for k, v in feats.items():
                if k in df.columns:
                    df.at[idx, k] = v
                else:
                    df[k] = np.nan
                    df.at[idx, k] = v
        except Exception:
            continue

    write_like_input(df, args.out)

if __name__ == "__main__":
    main()
