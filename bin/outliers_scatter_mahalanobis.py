#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outliers_scatter_mahalanobis.py
-------------------------------
Visualizes Train vs Test coverage using PCA(2D).
Colors points by Mahalanobis distance (distance to the center of the data).
Robust to missing 'is_outlier' column.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def fit_mahalanobis(X):
    """Calculates Mahalanobis distance for X."""
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    mean = np.mean(X, axis=0)
    diff = X - mean
    dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    return dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outlier_col", default="is_outlier")
    ap.add_argument("--only_cols", nargs="+", default=None)
    ap.add_argument("--basis", choices=["train", "combined"], default="combined")
    ap.add_argument("--outdir", default="out_viz")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_tr = pd.read_parquet(args.train)
    df_te = pd.read_parquet(args.test)

    # 1. Select Columns
    numeric_cols = list(df_tr.select_dtypes(include=[np.number]).columns)
    if args.only_cols:
        numeric_cols = [c for c in args.only_cols if c in numeric_cols]
    
    # Remove ID/Target-like columns from PCA
    exclude = ["row_uid", "fold", args.outlier_col, "is_outlier"]
    feats = [c for c in numeric_cols if c not in exclude]

    print(f"[Scatter] Using features for PCA: {feats}")
    if len(feats) < 2:
        print("[ERROR] Need at least 2 numeric features for PCA.")
        sys.exit(0)

    # 2. Prepare Matrix
    X_tr = df_tr[feats].dropna().values
    X_te = df_te[feats].dropna().values
    
    # 3. Outlier Flags (Robust check)
    # If column doesn't exist (already filtered), assume 0
    if args.outlier_col in df_tr.columns:
        out_tr = df_tr[args.outlier_col].fillna(0).astype(int).values
    else:
        out_tr = np.zeros(len(X_tr))
        
    if args.outlier_col in df_te.columns:
        out_te = df_te[args.outlier_col].fillna(0).astype(int).values
    else:
        out_te = np.zeros(len(X_te))

    # 4. Standardization & PCA
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    
    if args.basis == "train":
        scaler.fit(X_tr)
        X_tr_sc = scaler.transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        pca.fit(X_tr_sc)
    else:
        X_comb = np.vstack([X_tr, X_te])
        scaler.fit(X_comb)
        X_tr_sc = scaler.transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        pca.fit(np.vstack([X_tr_sc, X_te_sc]))

    P_tr = pca.transform(X_tr_sc)
    P_te = pca.transform(X_te_sc)

    # 5. Mahalanobis Distance (Color)
    # We calculate dist based on Train distribution usually
    # But here let's visualize distance to the global center
    # dists = fit_mahalanobis(np.vstack([P_tr, P_te])) # simplified on PCA space
    
    # 6. Plot
    plt.figure(figsize=(10, 8))
    
    # Train points
    plt.scatter(P_tr[:, 0], P_tr[:, 1], alpha=0.5, label='Train', s=15, c='blue', edgecolors='none')
    # Test points
    plt.scatter(P_te[:, 0], P_te[:, 1], alpha=0.5, label='Test', s=15, c='orange', edgecolors='none')
    
    # Highlight outliers (Red circles)
    if out_tr.sum() > 0:
        plt.scatter(P_tr[out_tr==1, 0], P_tr[out_tr==1, 1], s=50, facecolors='none', edgecolors='red', label='Train Outlier')
    
    if out_te.sum() > 0:
        plt.scatter(P_te[out_te==1, 0], P_te[out_te==1, 1], s=50, facecolors='none', edgecolors='magenta', label='Test Outlier')

    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    plt.title("PCA Space: Train vs Test Coverage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(args.outdir, "pca_scatter.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Scatter] Saved: {out_path}")

if __name__ == "__main__":
    main()