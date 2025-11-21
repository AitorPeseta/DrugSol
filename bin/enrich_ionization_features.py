#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
import re
import json
import requests

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# ============================================================================
# === IONIZACIÓN =============================================================
# ============================================================================

def split_acid_base_pattern(smarts_file):
    """Carga y separa los patrones SMARTS ácidos (A) y básicos (B)."""
    try:
        df_smarts = pd.read_csv(smarts_file, sep="\t")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: El archivo de patrones SMARTS '{smarts_file}' no se encontró."
        )
    df_smarts_acid = df_smarts[df_smarts.Acid_or_base == "A"].reset_index(drop=True)
    df_smarts_base = df_smarts[df_smarts.Acid_or_base == "B"].reset_index(drop=True)
    return df_smarts_acid, df_smarts_base


def unique_acid_match(matches):
    """Lógica especial para manejar átomos compartidos en grupos ácidos."""
    single_matches = list(set([m[0] for m in matches if len(m) == 1]))
    double_matches = [m for m in matches if len(m) == 2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches


def match_acid(df_smarts_acid, mol):
    """Encuentra los índices atómicos únicos de los sitios ácidos."""
    matches = []

    for row in df_smarts_acid.itertuples():
        try:
            smarts_str = row.SMARTS
            index_str = row.new_index
        except AttributeError:
            try:
                smarts_str = row[2]
                index_str = row[3]
            except IndexError:
                continue

        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None:
            continue

        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue

        if isinstance(index_str, str) and "," in index_str:
            index = [int(i) for i in index_str.split(",")]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
        else:
            try:
                index = int(index_str)
            except (ValueError, TypeError):
                continue

            for m in match:
                if index < len(m):
                    matches.append([m[index]])

    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)

    return list(set(matches_modify))


def match_base(df_smarts_base, mol):
    """Encuentra los índices atómicos únicos de los sitios básicos."""
    matches = []

    for row in df_smarts_base.itertuples():
        try:
            smarts_str = row.SMARTS
            index_str = row.new_index
        except AttributeError:
            try:
                smarts_str = row[2]
                index_str = row[3]
            except IndexError:
                continue

        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None:
            continue

        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue

        if isinstance(index_str, str) and "," in index_str:
            index = [int(i) for i in index_str.split(",")]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
        else:
            try:
                index = int(index_str)
            except (ValueError, TypeError):
                continue

            for m in match:
                if index < len(m):
                    matches.append([m[index]])

    unique_matches_tuples = unique_acid_match(matches)
    matches_modify = []
    for i in unique_matches_tuples:
        for j in i:
            matches_modify.append(j)

    return list(set(matches_modify))


def get_ionization_aid(mol, df_smarts_acid, df_smarts_base):
    if mol is None:
        return [], []

    mol = AllChem.AddHs(mol)

    acid_matches = match_acid(df_smarts_acid, mol)
    base_matches = match_base(df_smarts_base, mol)

    return acid_matches, base_matches


def calculate_ionization_features(smiles, df_smarts_acid, df_smarts_base):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return pd.Series({"n_ionizable": 0, "n_acid": 0, "n_base": 0})

    acid_idx, base_idx = get_ionization_aid(mol, df_smarts_acid, df_smarts_base)

    n_acid = len(acid_idx)
    n_base = len(base_idx)
    n_ionizable = n_acid + n_base

    return pd.Series({
        "n_ionizable": int(n_ionizable),
        "n_acid": int(n_acid),
        "n_base": int(n_base),
    })


# ============================================================================
# === FENOLES ================================================================
# ============================================================================

_PHENOL_SMARTS_STRICT = Chem.MolFromSmarts("c[OX2H]")
_PHENOL_SMARTS_LOOSE = Chem.MolFromSmarts("[CX3,cX3][OX2H]")


def phenol_features(mol: Chem.Mol):
    if mol is None:
        return dict(n_phenol=0, has_phenol=0, n_phenol_like=0, phenol_like=0)

    def count_unique_O(smarts):
        if smarts is None:
            return 0
        matches = mol.GetSubstructMatches(smarts)
        if not matches:
            return 0
        o_idxs = {m[1] for m in matches if len(m) > 1}
        return len(o_idxs)

    n_strict = count_unique_O(_PHENOL_SMARTS_STRICT)
    n_loose = count_unique_O(_PHENOL_SMARTS_LOOSE)

    return dict(
        n_phenol=int(n_strict),
        has_phenol=int(n_strict > 0),
        n_phenol_like=int(n_loose),
        phenol_like=int(n_loose > 0),
    )


# ============================================================================
# === pKa vía MolGpKa API ====================================================
# ============================================================================

DEFAULT_PKA_URL = "http://xundrug.cn:5001/modules/upload0/"


def predict_pka_molgpka(smiles: str, api_url: str, token: str):
    """
    Llama al servidor MolGpKa y devuelve res_json['gen_datas'].

    ¡Ojo! No hay validación fuerte del esquema de salida, sólo se comprueba
    que status == 200 y ifjson == 1.
    """
    if not smiles:
        return None

    param = {"Smiles": ("tmg", smiles)}
    headers = {"token": token}

    resp = requests.post(url=api_url, files=param, headers=headers, timeout=30)
    resp.raise_for_status()

    try:
        jsonbool = int(resp.headers.get("ifjson", "0"))
    except ValueError:
        jsonbool = 0

    if jsonbool != 1:
        raise RuntimeError("MolGpKa: cabecera ifjson != 1")

    res_json = resp.json()
    if int(res_json.get("status", 0)) != 200:
        raise RuntimeError(f"MolGpKa: status != 200 ({res_json.get('status')})")

    return res_json.get("gen_datas")

def summarize_pka(val):
    """
    val: puede ser dict o string JSON con estructura:
      {"Acid": {"9": "2.78", "14": "3.11"}, "Base": {...}, "MolBlock": "..."}
    Devuelve un dict con:
      pka_acidic_min, pka_acidic_max, pka_acidic_n,
      pka_basic_min,  pka_basic_max,  pka_basic_n
    """
    def _empty():
        return dict(
            pka_acidic_min=0,
            pka_acidic_max=0,
            pka_acidic_n=0,
            pka_basic_min=0,
            pka_basic_max=0,
            pka_basic_n=0,
        )

    # Nada o NaN → todo vacío
    if val is None:
        return _empty()
    if isinstance(val, float) and math.isnan(val):
        return _empty()

    # Asegurarnos de tener un dict
    if isinstance(val, dict):
        obj = val
    else:
        s = str(val).strip()
        if not s:
            return _empty()
        try:
            obj = json.loads(s)
        except Exception:
            # Si el JSON es inválido → vacío
            return _empty()

    acid = obj.get("Acid") or {}
    base = obj.get("Base") or {}

    def _stats(d):
        vals = []
        for v in d.values():
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            return (0, 0, 0)
        arr = np.asarray(vals, dtype=float)
        return (float(np.nanmin(arr)), float(np.nanmax(arr)), int(len(arr)))

    a_min, a_max, a_n = _stats(acid)
    b_min, b_max, b_n = _stats(base)

    return dict(
        pka_acidic_min=a_min,
        pka_acidic_max=a_max,
        pka_acidic_n=a_n,
        pka_basic_min=b_min,
        pka_basic_max=b_max,
        pka_basic_n=b_n,
    )




# ============================================================================
# === I/O ====================================================================
# ============================================================================

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


# ============================================================================
# === MAIN ===================================================================
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        "Añade features de ionización + pKa (MolGpKa) + fenoles al dataset"
    )
    ap.add_argument("--in", dest="inp", required=True,
                    help="Ruta al archivo de entrada (CSV, TXT, Parquet).")
    ap.add_argument("--out", dest="out", required=True,
                    help="Ruta al archivo de salida.")
    ap.add_argument("--smiles-col", default="smiles_neutral",
                    help="Nombre de la columna que contiene los SMILES.")
    ap.add_argument("--smarts", required=True,
                    help="Ruta al archivo 'smarts_pattern_ionized.txt'.")
    ap.add_argument("--pka-api-url", default=None,
                    help="URL de la API de pKa (MolGpKa). Si no se da, se usa el default.")
    ap.add_argument("--pka-token", default=None,
                    help="Token para la API de pKa (MolGpKa). Si falta, no se calcula pKa.")

    args = ap.parse_args()

    df = read_any(args.inp).copy()
    if args.smiles_col not in df.columns:
        raise SystemExit(f"Falta columna {args.smiles_col}")

    # --- Pre-carga de patrones SMARTS ---
    print("Cargando patrones SMARTS.")
    df_smarts_acid, df_smarts_base = split_acid_base_pattern(args.smarts)

    # --- Features de ionización ---
    print("Calculando features de ionización (n_ionizable, n_acid, n_base).")
    new_features = df[args.smiles_col].apply(
        lambda x: calculate_ionization_features(x, df_smarts_acid, df_smarts_base)
    )
    df = pd.concat([df, new_features], axis=1)
    df["n_ionizable"] = df["n_ionizable"].astype(int)
    df["n_acid"] = df["n_acid"].astype(int)
    df["n_base"] = df["n_base"].astype(int)

    # --- Fenoles ---
    print("Calculando features de fenoles.")
    df["Mol"] = df[args.smiles_col].apply(Chem.MolFromSmiles)
    phenol_results = df["Mol"].apply(phenol_features).apply(pd.Series)
    df = pd.concat([df, phenol_results], axis=1)
    df = df.drop(columns=["Mol"])

    # --- pKa vía MolGpKa API ---
    if args.pka_token:
        api_url = args.pka_api_url or DEFAULT_PKA_URL
        print(f"Calculando pKa con MolGpKa API ({api_url}).")
        cache = {}
        pka_rows = []

        for smi in df[args.smiles_col]:
            smi_key = smi if isinstance(smi, str) else str(smi)
            if smi_key not in cache:
                try:
                    gen_datas = predict_pka_molgpka(smi_key, api_url, args.pka_token)
                except Exception as e:
                    print(f"[WARN] Error al obtener pKa para SMILES '{smi_key}': {e}")
                    gen_datas = None
                cache[smi_key] = summarize_pka(gen_datas)
            pka_rows.append(cache[smi_key])

        if "pka_raw_json" in df.columns:
            print("Resumiendo pKa desde pka_raw_json...")
            pka_summary = df["pka_raw_json"].apply(summarize_pka_json).apply(pd.Series)
            df = pd.concat([df, pka_summary], axis=1)
        else:
            print("No hay columna 'pka_raw_json'; no se resumen pKas.")

        pka_df = pd.DataFrame(pka_rows)
        df = pd.concat([df.reset_index(drop=True), pka_df.reset_index(drop=True)], axis=1)
    else:
        print("No se ha proporcionado --pka-token. Se omite cálculo de pKa (MolGpKa).")

    # --- Guardar salida ---
    write_like_input(df, args.out)
    print(f"Proceso completado. Resultados guardados en {args.out}")


if __name__ == "__main__":
    main()
