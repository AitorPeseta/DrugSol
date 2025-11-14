#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

# RDKit: solo para descriptores
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------- I/O helpers ----------------

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

# ---------------- RDKit helpers ----------------

def safe_mol_from_smiles(smi):
    try:
        if smi is None or (isinstance(smi, float) and math.isnan(smi)):
            return None
        smi = str(smi).strip()
        if not smi:
            return None
        mol = Chem.MolFromSmiles(smi)
        return mol
    except Exception:
        return None

def compute_rdkit_basic(mol):
    """
    Devuelve dict con:
      TPSA, logP, HBD, HBA, FractionCSP3, MW
    Si mol es None, devuelve todo NaN.
    """
    if mol is None:
        return {
            "TPSA": np.nan,
            "logP": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "FractionCSP3": np.nan,
            "MW": np.nan,
        }
    try:
        # TPSA
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        # logP (Crippen)
        logp = Descriptors.MolLogP(mol)
        # Donors/Acceptors (Lipinski-like)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        # FractionCSP3
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        # Peso molecular exacto “estándar” (no exact mass)
        mw = Descriptors.MolWt(mol)

        return {
            "TPSA": float(tpsa),
            "logP": float(logp),
            "HBD": int(hbd),
            "HBA": int(hba),
            "FractionCSP3": float(fsp3),
            "MW": float(mw),
        }
    except Exception:
        return {
            "TPSA": np.nan,
            "logP": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "FractionCSP3": np.nan,
            "MW": np.nan,
        }

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser("Añade features RDKit básicas (TPSA, logP, HBD, HBA, FractionCSP3, MW)")
    ap.add_argument("--in", dest="inp", required=True, help="CSV/Parquet de entrada")
    ap.add_argument("--out", dest="out", required=True, help="CSV/Parquet de salida")
    ap.add_argument("--smiles-col", default="smiles_neutral", help="Columna con SMILES (por defecto: smiles_neutral)")
    args = ap.parse_args()

    df = read_any(args.inp).copy()
    if args.smiles_col not in df.columns:
        raise SystemExit(f"Falta columna {args.smiles_col}")

    # Pre-crear columnas (numéricas)
    target_cols = ["TPSA", "logP", "HBD", "HBA", "FractionCSP3", "MW"]
    for c in target_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Calcular fila a fila (robusto)
    smiles_iter = df[args.smiles_col].values
    for i, smi in enumerate(smiles_iter):
        mol = safe_mol_from_smiles(smi)
        vals = compute_rdkit_basic(mol)
        for k, v in vals.items():
            df.at[i, k] = v

    write_like_input(df, args.out)

if __name__ == "__main__":
    main()
