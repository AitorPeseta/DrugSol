#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make_features_mordred.py  (RDKit logP + Mordred + solvent OHE + features de InChIKey14; sin eliminar columnas originales)

import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen  # <- para MolLogP

# Mordred / MordredCommunity
try:
    from mordred import Calculator, descriptors
    _MORDRED_PKG = "mordred"
except Exception:
    try:
        from mordredcommunity import Calculator, descriptors
        _MORDRED_PKG = "mordredcommunity"
    except Exception:
        print("No se pudo importar ni 'mordred' ni 'mordredcommunity'. Instala uno de ellos.")
        sys.exit(1)


def smiles_to_mol(s):
    if pd.isna(s):
        return None
    try:
        return Chem.MolFromSmiles(str(s))
    except Exception:
        return None


def add_3d_conformer(mol, seed=42, max_iters=200, forcefield="uff", max_atoms=200):
    if mol is None:
        return None
    try:
        if mol.GetNumAtoms() >= max_atoms:
            return None
        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        if AllChem.EmbedMolecule(m, params) == -1:
            return None
        try:
            if forcefield.lower() == "mmff" and AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=int(max_iters))
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=int(max_iters))
        except Exception:
            pass
        return Chem.RemoveHs(m)
    except Exception:
        return None


def compute_mordred(df_mols, ignore_3d=True, nproc=1):
    calc = Calculator(descriptors, ignore_3D=ignore_3d)
    mord = calc.pandas(df_mols["mol"], nproc=nproc)
    mord = mord.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    mord.columns = [f"mordred__{c}" for c in mord.columns]
    return mord


def compute_rdkit_logp(mols: pd.Series) -> pd.Series:
    """Crippen logP; NaN si SMILES inválido."""
    def _lp(m):
        try:
            return float(Crippen.MolLogP(m)) if m is not None else np.nan
        except Exception:
            return np.nan
    s = mols.apply(_lp).astype("float32")
    s.name = "rdkit__logP"
    return s


def _stable_hash_to_bin(s: str, n_bins: int) -> int:
    import hashlib
    h = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)
    return h % n_bins


def inchikey14_hash_features(series: pd.Series, n_bins: int) -> pd.DataFrame:
    n = len(series)
    arr = np.zeros((n, n_bins), dtype=np.uint8)
    idx = series.fillna("").astype(str).apply(lambda x: _stable_hash_to_bin(x, n_bins)).to_numpy()
    arr[np.arange(n), idx] = 1
    cols = [f"ik14h_{i:03d}" for i in range(n_bins)]
    return pd.DataFrame(arr, index=series.index, columns=cols)


def inchikey14_frequency(series: pd.Series) -> pd.Series:
    vc = series.value_counts(dropna=False)
    freq = series.map(vc).astype("float32") / float(len(series))
    return freq.astype("float32").rename("ik14_freq")


def add_suffix_if_missing(path: str, suffix: str) -> str:
    return path if path.lower().endswith(suffix) else f"{path}{suffix}"


def main():
    ap = argparse.ArgumentParser(
        description=("Calcula RDKit logP + Mordred (2D/3D), solvent OHE e InChIKey14 (hash+freq). "
                     "No elimina columnas existentes; solo añade nuevas. Opcionalmente conserva SMILES.")
    )
    ap.add_argument("-i", "--input", required=True, help="Entrada .parquet o .csv")
    ap.add_argument("--smiles_neutral_col", default="smiles_neutral")
    ap.add_argument("--smiles_original_col", default="smiles_original")
    ap.add_argument("--smiles_source", choices=["neutral", "original"], default="neutral",
                    help="Cuál columna de SMILES usar para RDKit/Mordred y (si --keep-smiles) conservar en salida")
    ap.add_argument("--keep-smiles", action="store_true",
                    help="Conserva en la salida la columna de SMILES seleccionada por --smiles_source")
    ap.add_argument("--solvent_col", default="solvent")
    ap.add_argument("--inchikey_col", default="InChIKey14")
    ap.add_argument("--ik14_hash_bins", type=int, default=128)
    ap.add_argument("--keep_inchikey_as_group", action="store_true",
                    help="Si se activa, exporta columna 'groups' con InChIKey14 (string)")
    ap.add_argument("--out_parquet", default="features_mordred.parquet")
    ap.add_argument("--save_csv", action="store_true")

    ap.add_argument("--include_3d", action="store_true")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--seed_3d", type=int, default=42)
    ap.add_argument("--ff", choices=["uff", "mmff"], default="uff")
    ap.add_argument("--max_atoms_3d", type=int, default=200)
    ap.add_argument("--max_iters_3d", type=int, default=200)
    ap.add_argument("--fail-on-invalid", action="store_true")
    ap.add_argument("--mordred-keep", nargs="+", default=None,
                    help="Lista de descriptores Mordred a conservar (sin el prefijo)")

    args = ap.parse_args()
    if not os.path.exists(args.input):
        print(f"No se encontró: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Carga
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        print("El archivo de entrada debe ser .parquet o .csv", file=sys.stderr)
        sys.exit(1)

    # Determinar columna de SMILES a usar
    chosen_smiles_col = args.smiles_neutral_col if args.smiles_source == "neutral" else args.smiles_original_col
    if chosen_smiles_col not in df.columns:
        print(f"Falta la columna de SMILES seleccionada: {chosen_smiles_col}", file=sys.stderr)
        sys.exit(1)

    # Crear moléculas
    mols = df[chosen_smiles_col].apply(smiles_to_mol)
    valid_mask = mols.notna()
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        msg = f"[AVISO] {n_invalid} SMILES inválidos; esas filas tendrán NaNs en RDKit/Mordred."
        if args.fail_on_invalid:
            raise SystemExit("[ERROR] " + msg)
        else:
            print(msg, file=sys.stderr)

    # 3D opcional (antes de Mordred si se usa 3D)
    if args.include_3d:
        def _add3d(m):
            if m is None:
                return None
            return add_3d_conformer(m, seed=args.seed_3d, max_iters=args.max_iters_3d,
                                    forcefield=args.ff, max_atoms=args.max_atoms_3d)
        mols_for_mordred = mols.apply(_add3d)
    else:
        mols_for_mordred = mols

    # -------- NUEVO: RDKit logP (primera columna añadida) --------
    rdkit_logp = compute_rdkit_logp(mols)  # usa la versión 2D estándar (Crippen)

    # Mordred
    df_mols = pd.DataFrame({"mol": mols_for_mordred})
    df_mols_valid = df_mols[valid_mask]
    if len(df_mols_valid) == 0:
        mord = pd.DataFrame(index=df_mols.index)
    else:
        mord_valid = compute_mordred(df_mols_valid, ignore_3d=not args.include_3d, nproc=args.nproc)
        mord_valid.index = df_mols_valid.index
        mord = mord_valid.reindex(df_mols.index)

    if args.mordred_keep:
        wanted = set([str(x) for x in args.mordred_keep])
        mord_cols = mord.columns.tolist()
        base = [c.replace("mordred__", "", 1) if c.startswith("mordred__") else c for c in mord_cols]
        keep_mask = [b in wanted for b in base]
        mord = mord.loc[:, keep_mask]

    # One-hot de solvente (no se elimina la columna original)
    solvent_ohe = pd.get_dummies(
        df[args.solvent_col].astype(str) if args.solvent_col in df.columns else pd.Series([], dtype=str, index=df.index),
        prefix="solv",
        dummy_na=False
    ).astype("int8")

    # InChIKey14 hashing + frecuencia (no se elimina la columna original)
    ik_col = args.inchikey_col
    has_ik = ik_col in df.columns
    if has_ik:
        ik_series = df[ik_col].astype(str)
        ik_hash = inchikey14_hash_features(ik_series, args.ik14_hash_bins)
        ik_freq = inchikey14_frequency(ik_series)
    else:
        print(f"[AVISO] no se encontró '{ik_col}'; sin features IK14.", file=sys.stderr)
        ik_hash = pd.DataFrame(index=df.index)
        ik_freq = pd.Series(index=df.index, dtype="float32", name="ik14_freq")

    # Grupos (opcional)
    extra_cols = []
    if args.keep_inchikey_as_group and has_ik:
        extra_cols.append(df[ik_col].astype(str).rename("groups"))

    # Construir salida: **NO** eliminar columnas originales; solo añadir nuevas.
    # Orden de nuevas columnas: rdkit__logP, luego Mordred, luego solvent OHE, IK hash/freq, y opcionales.
    parts = [df.copy(), rdkit_logp.to_frame()]
    if not mord.empty:
        parts.append(mord)
    if not solvent_ohe.empty:
        parts.append(solvent_ohe)
    if not ik_hash.empty:
        parts.append(ik_hash)
    if ik_freq.notna().any() or (len(ik_freq) == len(df)):
        parts.append(ik_freq.to_frame())
    for ec in extra_cols:
        parts.append(ec.to_frame())

    feats = pd.concat(parts, axis=1)

    # Conservar SMILES seleccionados si se pidió (ya está en df; aquí solo garantizamos que existe)
    if args.keep_smiles and chosen_smiles_col in feats.columns:
        pass  # ya está

    out_parquet = add_suffix_if_missing(args.out_parquet, ".parquet")
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_parquet, index=False)
    print(f"[OK] Guardado Parquet: {out_parquet}")
    if args.save_csv:
        out_csv = os.path.splitext(out_parquet)[0] + ".csv"
        feats.to_csv(out_csv, index=False)
        print(f"[OK] Guardado CSV: {out_csv}")


if __name__ == "__main__":
    main()
