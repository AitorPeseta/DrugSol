#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def _to_mol(s: str):
    if pd.isna(s):
        return None
    try:
        return Chem.MolFromSmiles(str(s))
    except Exception:
        return None


def _cluster_labels_from_bitvects(bitvects, cutoff: float):
    """
    Devuelve un vector con IDs de cluster para los índices válidos.
    `bitvects` debe ser una lista de RDKit ExplicitBitVect (solo válidos).
    """
    n = len(bitvects)
    if n == 0:
        return np.array([], dtype=int)

    # Distancias condensadas para Butina: 1 - Tanimoto
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, nPts=n, distThresh=1.0 - cutoff, isDistData=True)
    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels


def main():
    ap = argparse.ArgumentParser("Genera ECFP4 como columnas de bits + cluster Butina por Tanimoto (añadiendo sin borrar).")
    ap.add_argument("--input", "-i", required=True, help="Entrada .parquet o .csv")
    ap.add_argument("--smiles-col", default="smiles_neutral", help="Columna con SMILES (default: smiles_neutral)")
    ap.add_argument("--n-bits", type=int, default=2048, help="Tamaño del fingerprint (default: 2048)")
    ap.add_argument("--radius", type=int, default=2, help="Radio para Morgan/ECFP (default: 2 ≡ ECFP4)")
    ap.add_argument("--cluster-cutoff", type=float, default=0.7, help="Umbral Tanimoto para Butina (default: 0.7)")
    ap.add_argument("--bits-prefix", default="ecfp4_b", help="Prefijo para las columnas de bits (default: ecfp4_b)")
    ap.add_argument("--out-parquet", default="with_ecfp4_bits.parquet", help="Salida Parquet")
    ap.add_argument("--save-csv", action="store_true", help="Guardar también CSV")
    args = ap.parse_args()

    # --- carga ---
    if not os.path.exists(args.input):
        print(f"[ERROR] No se encontró: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        print("[ERROR] La entrada debe ser .parquet o .csv", file=sys.stderr)
        sys.exit(1)

    assert args.smiles_col in df.columns, f"Falta columna SMILES: {args.smiles_col}"

    # --- RDKit mols y fingerprints ---
    mols = df[args.smiles_col].apply(_to_mol)
    valid_idx = mols.notna().to_numpy().nonzero()[0]

    # lista de bitvect (solo válidos)
    bitvects_valid = []
    for i in valid_idx:
        m = mols.iat[i]
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=int(args.radius), nBits=int(args.n_bits))
        bitvects_valid.append(bv)

    # volcamos bits a una matriz numpy (n_valid, n_bits) de uint8
    bits_valid = np.zeros((len(valid_idx), int(args.n_bits)), dtype=np.uint8)
    for row, bv in enumerate(bitvects_valid):
        # rellenamos bits_valid[row, :] in-place desde ExplicitBitVect
        DataStructs.ConvertToNumpyArray(bv, bits_valid[row, :])  # recomendado por RDKit
        # ConvertToNumpyArray devuelve 0/1 en el array destino

    # matriz completa (n, n_bits) con ceros para inválidos
    n = len(df)
    bits_full = np.zeros((n, int(args.n_bits)), dtype=np.uint8)
    bits_full[valid_idx, :] = bits_valid

    # dataframe de bits, un bit por columna
    bit_cols = [f"{args.bits_prefix}{i:04d}" for i in range(int(args.n_bits))]
    df_bits = pd.DataFrame(bits_full, columns=bit_cols, index=df.index)

    # --- clusters (solo válidos) ---
    lab_valid = _cluster_labels_from_bitvects(bitvects_valid, cutoff=float(args.cluster_cutoff))
    cluster_full = np.full(n, -1, dtype=int)
    cluster_full[valid_idx] = lab_valid

    # nombre de la columna de cluster con sufijo del cutoff (p. ej. 0p7)
    cutoff_tag = str(args.cluster_cutoff).replace(".", "p")
    cluster_col = f"cluster_ecfp4_{cutoff_tag}"

    out = pd.concat([df, df_bits], axis=1)
    out[cluster_col] = cluster_full

    # --- limpieza opcional (no dejar columna empaquetada si llegó de otra etapa)
    if "ecfp4_bits" in out.columns:
        out = out.drop(columns=["ecfp4_bits"])

    # --- guardar: siempre parquet; CSV opcional ---
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print(f"[OK] Guardado Parquet: {out_parquet}")

    if args.save_csv:
        out_csv = out_parquet.with_suffix(".csv")
        out.to_csv(out_csv, index=False)
        print(f"[OK] Guardado CSV: {out_csv}")



if __name__ == "__main__":
    main()
