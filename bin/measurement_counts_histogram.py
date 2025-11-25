#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measurement_counts_histogram.py
-------------------------------
Generates a histogram showing how many times each molecule appears in the dataset.
Useful for detecting redundancy or bias towards specific compounds.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

def freq_from_counts(counts: pd.Series) -> pd.Series:
    """
    Input: Series indexed by molecule ID containing the number of measurements.
    Output: Frequency series (how many molecules have N measurements).
    """
    if counts.empty:
        return pd.Series(dtype=int)
    return counts.value_counts().sort_index().astype(int)

def main():
    ap = argparse.ArgumentParser(description="Histogram of measurement counts per molecule.")
    ap.add_argument("--input", required=True, help="Input Parquet/CSV file")
    ap.add_argument("--id_col", required=True, help="Molecule Identifier Column (e.g., smiles, mol_id)")
    ap.add_argument("--outdir", default="measurements_out", help="Output directory")
    ap.add_argument("--max_bin", type=int, default=None, help="Max X-axis value (cutoff)")
    ap.add_argument("--as-perc", dest="as_perc", action="store_true", help="Show percentage instead of absolute count")
    
    # Ticks control
    ap.add_argument("--xminor", type=float, default=1.0, help="Minor tick step for X axis")
    ap.add_argument("--grid-alpha", type=float, default=0.3, help="Grid transparency (0-1)")
    
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load Data
    try:
        if args.input.endswith(".parquet"):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to read input: {e}")
        return
    
    if args.id_col not in df.columns:
        print(f"[ERROR] Column '{args.id_col}' not found in dataset.")
        return

    # 2. Count measurements per molecule
    # Group by ID and count occurrences
    counts = df.groupby(args.id_col).size().rename("count_per_mol")
    counts.to_csv(os.path.join(args.outdir, "measurement_counts.csv"))

    # 3. Calculate Frequencies (Distribution of counts)
    freq = freq_from_counts(counts)
    
    # Filter if max_bin is set
    if args.max_bin is not None:
        freq = freq[freq.index <= args.max_bin]

    # 4. Prepare Plot Data
    freq_df = freq.reset_index()
    freq_df.columns = ["n_measurements", "n_molecules"]
    
    if args.as_perc:
        total = freq_df["n_molecules"].sum()
        y_vals = (freq_df["n_molecules"] / total * 100.0).round(2)
        ylabel = "Percentage of Molecules (%)"
    else:
        y_vals = freq_df["n_molecules"].values
        ylabel = "Count of Molecules"

    x_vals = freq_df["n_measurements"].values
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(x_vals, y_vals, width=0.8, color="#008080", zorder=3)
    
    plt.xlabel("Number of Measurements per Molecule")
    plt.ylabel(ylabel)
    plt.title("Redundancy Analysis")
    
    # --- AXIS CORRECTION ---
    ax = plt.gca()
    
    # Define exact limits based on data
    if len(x_vals) > 0:
        plt.xlim(min(x_vals) - 1, max(x_vals) + 1)

    # Use MaxNLocator to enforce integer ticks and avoid overcrowding
    # 'integer=True' ensures we don't see decimals (e.g. 1.5 measurements)
    # 'nbins' prevents label overlap on very long tails
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=30))

    # Minor ticks (optional)
    if args.xminor > 0:
        ax.xaxis.set_minor_locator(MultipleLocator(args.xminor))
    
    # Grid settings
    plt.grid(which="major", axis="y", linestyle="-", alpha=args.grid_alpha, zorder=0)
    plt.grid(which="minor", axis="x", linestyle=":", alpha=args.grid_alpha * 0.5, zorder=0)

    plt.tight_layout()
    
    out_png = os.path.join(args.outdir, "measurements_hist.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    print(f"[OK] Redundancy plot saved: {out_png}")

if __name__ == "__main__":
    main()