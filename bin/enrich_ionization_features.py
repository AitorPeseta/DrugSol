#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
import re  
import requests  

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# === FUNCIONES DE IONIZACIÓN (Corregidas para la lógica de índices) ========================

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

    # El DataFrame tiene las columnas:
    # [0]=Index, [1]=Substructure, [2]=SMARTS, [3]=new_index, [4]=Index, [5]=Acid_or_base
    for row in df_smarts_acid.itertuples():
        # Acceso seguro a columnas (usa new_index, que es row[3])
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

        # Lógica de parsing de índice (usa el índice 0-base en la tupla 'match')
        if isinstance(index_str, str) and "," in index_str:
            index = [int(i) for i in index_str.split(",")]

            for m in match:
                # El Código A solo usa los dos primeros índices del patrón (index[0], index[1])
                matches.append([m[index[0]], m[index[1]]])
        else:
            try:
                index = int(index_str)
            except (ValueError, TypeError):
                continue

            for m in match:
                # La corrección incluye la verificación de rango para evitar IndexError
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
        # Acceso seguro a columnas (usa new_index, que es row[3])
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

        # Lógica de parsing de índice (usa el índice 0-base en la tupla 'match')
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

    # El conteo es simplemente la longitud de las listas de índices únicos devueltas
    n_acid = len(acid_idx)
    n_base = len(base_idx)
    n_ionizable = n_acid + n_base

    return pd.Series({
        "n_ionizable": int(n_ionizable),
        "n_acid": int(n_acid),
        "n_base": int(n_base),
    })


# ----------------------------------------------------------------------------
# === FENOLES (Corregido para incluir enoles) ===================

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
        # El átomo de Oxígeno siempre está en el índice 1 del match para X[OX2H]
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


# ----------------------------------------------------------------------------
# === PubChem: Melting Point por InChIKey (añadido) =========================

def _get_pubchem_cid_from_inchikey(inchikey: str):
    """
    Devuelve el primer CID asociado al InChIKey en PubChem.
    Si algo falla, devuelve None.
    """
    if not inchikey:
        return None
    inchikey = str(inchikey).strip()
    if not inchikey:
        return None

    try:
        url_cid = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"inchikey/{inchikey}/cids/JSON"
        )
        r = requests.get(url_cid, timeout=10)
        r.raise_for_status()
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None
        return int(cids[0])
    except Exception:
        return None


def _parse_mp_string_to_celsius(mp_str: str):
    if not mp_str:
        return 0
    s = str(mp_str)
    nums = re.findall(r"(-?\d+\.?\d*)", s)
    if not nums:
        return 0
    vals = [float(x) for x in nums]
    if not vals:
        return 0
    val = sum(vals) / len(vals)
    if re.search(r"\bK\b|kelvin", s, flags=re.IGNORECASE):
        val = val - 273.15
    return float(val)



def get_pubchem_melting_point_from_inchikey(inchikey: str):
    """
    Devuelve el melting point (ºC) desde PubChem para un InChIKey dado.
    Si no encuentra nada o hay error, devuelve Nan.
    """
    cid = _get_pubchem_cid_from_inchikey(inchikey)
    if cid is None:
        return np.nan

    try:
        url_view = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/"
            f"compound/{cid}/JSON"
        )
        r = requests.get(url_view, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return np.nan

    melting_points_strings = []

    def walk(section):
        if isinstance(section, dict):
            if section.get("TOCHeading") == "Experimental Properties":
                for s in section.get("Section", []):
                    if s.get("TOCHeading") == "Melting Point":
                        for info in s.get("Information", []):
                            val = info.get("Value", {}) or {}
                            for swm in val.get("StringWithMarkup", []):
                                melting_points_strings.append(swm.get("String", ""))
            for v in section.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(section, list):
            for x in section:
                walk(x)

    walk(data)

    if not melting_points_strings:
        return np.nan

    mp_val = _parse_mp_string_to_celsius(melting_points_strings[0])
    return mp_val


# ============== I/O Y MAIN (Funciones auxiliares) ==============

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
    ap = argparse.ArgumentParser("Añade features de ionización+pKa + fenoles (numéricos) al dataset")
    ap.add_argument("--in", dest="inp", required=True, help="Ruta al archivo de entrada (CSV, TXT, Parquet).")
    ap.add_argument("--out", dest="out", required=True, help="Ruta al archivo de salida.")
    ap.add_argument("--smiles-col", default="smiles_neutral", help="Nombre de la columna que contiene los SMILES.")
    ap.add_argument("--smarts", required=True, help="Ruta al archivo 'smarts_pattern_ionized.txt'.")
    ap.add_argument("--pka-api-url", default=None, help="URL de la API de pKa (opcional).")
    ap.add_argument("--pka-token", default=None, help="Token para la API de pKa (opcional).")
    # NUEVO: columna con InChIKey para poder bajar mp de PubChem
    ap.add_argument("--inchikey-col", default=None, help="Nombre de la columna con InChIKey (para bajar mp de PubChem).")
    args = ap.parse_args()

    df = read_any(args.inp).copy()
    if args.smiles_col not in df.columns:
        raise SystemExit(f"Falta columna {args.smiles_col}")

    # --- Pre-carga de patrones SMARTS ---
    print("Cargando patrones SMARTS.")
    df_smarts_acid, df_smarts_base = split_acid_base_pattern(args.smarts)

    # --- Cálculo de Features de Ionización ---
    print("Calculando features de ionización (n_ionizable, n_acid, n_base).")

    new_features = df[args.smiles_col].apply(
        lambda x: calculate_ionization_features(x, df_smarts_acid, df_smarts_base)
    )
    df = pd.concat([df, new_features], axis=1)

    df["n_ionizable"] = df["n_ionizable"].astype(int)
    df["n_acid"] = df["n_acid"].astype(int)
    df["n_base"] = df["n_base"].astype(int)

    # --- Cálculo de Features de Fenoles ---
    print("Calculando features de fenoles.")
    df["Mol"] = df[args.smiles_col].apply(Chem.MolFromSmiles)

    phenol_results = df["Mol"].apply(phenol_features).apply(pd.Series)
    df = pd.concat([df, phenol_results], axis=1)

    df = df.drop(columns=["Mol"])

    # --- Cálculo de Melting Point (mp) desde PubChem ---
    inchikey_col = args.inchikey_col
    has_inchikey = bool(inchikey_col) and (inchikey_col in df.columns)

    if has_inchikey:
        print("Calculando puntos de fusión (mp) desde PubChem.")
        def _safe_mp(val):
            if val is None:
                return np.nan
            if isinstance(val, float) and math.isnan(val):
                return np.nan
            try:
                return float(get_pubchem_melting_point_from_inchikey(str(val)))
            except Exception:
                return np.nan

        df["mp"] = df[inchikey_col].apply(_safe_mp)
    else:
        # Si no hay columna de InChIKey, no hacemos nada; por si acaso no pisamos una 'mp' existente
        print("No se ha proporcionado --inchikey-col o la columna no existe en el dataframe. No se calculará mp.")

    # --- Guardar el resultado ---
    write_like_input(df, args.out)
    print(f"Proceso completado. Resultados guardados en {args.out}")


if __name__ == "__main__":
    main()
