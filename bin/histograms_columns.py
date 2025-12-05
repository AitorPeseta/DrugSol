#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
histograms_columns.py
---------------------
Generates comparative histograms (Train vs Test).
Features:
- Auto-detection of Temperature units (K -> C).
- Overlay mode for distribution checks.
- On-the-fly calculation of QED if missing.
- Smart alias resolution (MW -> rdkit__MW).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Importaciones condicionales de RDKit para cálculo on-the-fly
try:
    from rdkit import Chem
    from rdkit.Chem import QED
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

def calculate_qed_on_the_fly(df):
    """Calcula QED desde SMILES si RDKit está disponible."""
    if not _HAS_RDKIT:
        print("[WARN] RDKit no instalado. No se puede calcular QED al vuelo.")
        return df
        
    # Buscar columna de smiles
    smiles_col = None
    candidates = ["smiles_neutral", "smiles_original", "smiles", "SMILES"]
    for c in candidates:
        if c in df.columns:
            smiles_col = c
            break
            
    if not smiles_col:
        print("[WARN] No se encontró columna de SMILES. No se puede calcular QED.")
        return df

    print(f"[Hist] Calculando QED on-the-fly usando '{smiles_col}'...")
    
    def _get_qed(s):
        if pd.isna(s): return np.nan
        m = Chem.MolFromSmiles(str(s))
        return QED.qed(m) if m else np.nan

    df["qed"] = df[smiles_col].apply(_get_qed)
    return df

def pick_columns(df, cols_cli):
    wanted = set(cols_cli or [])
    
    # --- FIX: Alias inteligente para MW ---
    # Si piden "MW" pero no existe, buscamos variantes comunes
    if "MW" in wanted and "MW" not in df.columns:
        if "rdkit__MW" in df.columns:
            wanted.remove("MW")
            wanted.add("rdkit__MW")
            print("[Hist] Usando 'rdkit__MW' para 'MW'")
        elif "mordred__MW" in df.columns:
            wanted.remove("MW")
            wanted.add("mordred__MW")
            print("[Hist] Usando 'mordred__MW' para 'MW'")
    # --------------------------------------
    
    # Si piden 'qed' y no está, intentamos calcularlo
    if "qed" in wanted and "qed" not in df.columns:
        df = calculate_qed_on_the_fly(df)
        
    # Si 'qed' está presente (calculado o nativo) y no estaba en wanted, lo dejamos estar
    # (solo lo añadimos a wanted si el usuario lo pidió explícitamente)

    existing = set(df.columns)
    selected = sorted(list(wanted & existing))
    missing = sorted(list(wanted - existing))
    
    if missing:
        print(f"[WARN] Columns not found in dataset: {missing}", file=sys.stderr)
        
    return selected, df

def convert_temp_k_to_c(colname, arr):
    """Heuristic to convert Kelvin to Celsius for plotting."""
    lower = colname.lower()
    if "temp" in lower and (arr.mean() > 150): # Simple heuristic
        return arr - 273.15, f"{colname} (°C)"
    return arr, colname

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="hist_out")
    ap.add_argument("--cols", nargs="+", required=True)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--overlay", action="store_true", help="Overlay Train/Test")
    ap.add_argument("--density", action="store_true", help="Normalize histograms (density)")
    ap.add_argument("--round-step", type=float, default=None)
    
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_tr = pd.read_parquet(args.train)
    df_te = pd.read_parquet(args.test)

    # Seleccionar y calcular columnas (incluyendo alias MW y QED)
    cols, df_tr = pick_columns(df_tr, args.cols)
    # Aplicar la misma lógica al test (importante para que QED se calcule si falta)
    _, df_te = pick_columns(df_te, args.cols)

    if not cols:
        print("[ERROR] No valid columns found to plot.")
        sys.exit(0)

    for c in cols:
        if c not in df_te.columns: continue
        
        s_tr = pd.to_numeric(df_tr[c], errors='coerce').dropna().values
        s_te = pd.to_numeric(df_te[c], errors='coerce').dropna().values
        
        # Convert Temp if needed
        s_tr, label_tr = convert_temp_k_to_c(c, s_tr)
        s_te, label_te = convert_temp_k_to_c(c, s_te)
        xlabel = label_tr

        print(f"[{c}] Train: {len(s_tr)}, Test: {len(s_te)}")

        plt.figure(figsize=(8, 5))
        
        all_data = np.concatenate([s_tr, s_te])
        if len(all_data) == 0: continue
        
        xmin, xmax = all_data.min(), all_data.max()
        
        # Configuración de etiquetas bonitas
        if "qed" in c.lower():
            xmin, xmax = 0.0, 1.0
            xlabel = "QED (Drug-likeness)"
        elif "mw" in c.lower():
            xlabel = "Molecular Weight (Da)"

        if args.overlay:
            plt.hist(s_tr, bins=args.bins, range=(xmin, xmax), alpha=0.5, density=args.density, label="Train", color='blue')
            plt.hist(s_te, bins=args.bins, range=(xmin, xmax), alpha=0.5, density=args.density, label="Test", color='orange')
            plt.ylabel("Density" if args.density else "Count")
            plt.legend()
        else:
            plt.hist(all_data, bins=args.bins, color='gray', alpha=0.7)
            plt.ylabel("Count")

        # Usamos el nombre de columna real para el título, pero la etiqueta bonita para el eje
        plt.title(f"Distribution: {c}")
        plt.xlabel(xlabel)
        plt.grid(True, alpha=0.3)
        
        # Limpieza del nombre de archivo (quita rdkit__ para que quede limpio)
        clean_name = c.replace("rdkit__", "").replace("mordred__", "")
        out_path = os.path.join(args.outdir, f"hist_{clean_name}.png")
        
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()