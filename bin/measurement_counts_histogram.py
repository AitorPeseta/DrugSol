#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--id_col", required=True)
    ap.add_argument("--outdir", default="measurements_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.input) if args.input.endswith('.parquet') else pd.read_csv(args.input)
    
    if args.id_col not in df.columns:
        print(f"[ERROR] Column {args.id_col} not found.")
        sys.exit(1)

    # Count occurrences
    counts = df[args.id_col].value_counts()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=range(1, counts.max()+2), align='left', rwidth=0.8, color='teal')
    plt.xlabel("Number of Measurements per Molecule")
    plt.ylabel("Count of Molecules")
    plt.title("Redundancy Analysis")
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(1, min(20, counts.max())+1)) # Show ticks for first 20
    
    out_png = os.path.join(args.outdir, "measurements_hist.png")
    plt.savefig(out_png, dpi=150)
    print(f"[Counts] Saved: {out_png}")
    
    # Save CSV stats
    counts.value_counts().sort_index().to_csv(os.path.join(args.outdir, "counts_summary.csv"), header=["n_molecules"])

if __name__ == "__main__":
    main()