#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_features.py
------------------
Advanced feature selection pipeline:
1. Removes Constant and Near-Zero Variance (NZV) columns.
2. Clusters highly correlated features (graph-based).
3. Selects one representative per cluster using LightGBM/XGBoost Feature Importance (Gain).
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd

# Optional LightGBM/XGBoost support
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def drop_constant_and_nzv(df: pd.DataFrame, nzv_thresh: float = 0.01) -> pd.DataFrame:
    """Drops constant and near-zero variance numerical columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    keep_cols = []
    
    for col in numeric_df.columns:
        series = numeric_df[col]
        # Constant check
        if series.nunique(dropna=True) <= 1:
            continue
            
        # NZV Check (dominance ratio)
        vc = series.value_counts(normalize=True, dropna=True)
        if not vc.empty and vc.iloc[0] >= (1.0 - nzv_thresh):
            continue
            
        keep_cols.append(col)
        
    return numeric_df[keep_cols]

def correlation_clusters(df: pd.DataFrame, corr_thresh: float) -> List[List[str]]:
    """
    Groups features into clusters based on absolute correlation > threshold.
    Returns a list of lists (clusters).
    """
    if df.shape[1] <= 1:
        return [df.columns.tolist()]
        
    # Calculate correlation matrix
    corr_matrix = df.corr().abs().fillna(0.0)
    cols = corr_matrix.columns.tolist()
    
    # Build adjacency graph
    adj: Dict[str, Set[str]] = {c: set() for c in cols}
    for i, c1 in enumerate(cols):
        for j in range(i+1, len(cols)):
            c2 = cols[j]
            if corr_matrix.iloc[i, j] >= corr_thresh:
                adj[c1].add(c2)
                adj[c2].add(c1)
                
    # Find connected components (clusters)
    visited = set()
    clusters = []
    for node in cols:
        if node in visited: continue
        
        component = []
        stack = [node]
        visited.add(node)
        
        while stack:
            curr = stack.pop()
            component.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        clusters.append(sorted(component))
        
    return clusters

def get_feature_importance_gain(X: pd.DataFrame, y: pd.Series, algo="lgbm") -> pd.Series:
    """Calculates feature importance (Gain) using a quick GBM model."""
    if y is None or y.isna().all():
        # Fallback to variance
        return X.var() / (X.var().sum() or 1.0)

    # LightGBM
    if algo == "lgbm" and _HAS_LGBM:
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        params = {
            "objective": "regression", "metric": "rmse", "verbosity": -1,
            "num_leaves": 31, "learning_rate": 0.05, "seed": 42
        }
        model = lgb.train(params, dtrain, num_boost_round=100)
        gain = model.feature_importance(importance_type="gain")
        return pd.Series(gain, index=X.columns)

    # XGBoost
    if algo == "xgb" and _HAS_XGB:
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=1)
        model.fit(X, y)
        gain = model.feature_importances_ # XGB uses gain by default in sklearn API usually
        return pd.Series(gain, index=X.columns)

    # Fallback
    return X.var()

def select_medoids(X: pd.DataFrame, y: pd.Series, clusters: List[List[str]], algo="lgbm") -> List[str]:
    """Selects the 'best' feature from each correlated cluster based on Gain."""
    selected_features = []
    
    # Calculate global importance once
    importances = get_feature_importance_gain(X, y, algo)
    
    for cluster in clusters:
        if len(cluster) == 1:
            selected_features.append(cluster[0])
            continue
            
        # Pick feature with max importance in this cluster
        best_feat = importances.reindex(cluster).fillna(0.0).idxmax()
        selected_features.append(best_feat)
        
    return selected_features

def main():
    ap = argparse.ArgumentParser(description="Filter Features by Correlation & Importance.")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--target", default="logS", help="Target column for importance calculation.")
    ap.add_argument("--mordred-prefix", default="mordred__", help="Prefix of columns to filter.")
    ap.add_argument("--corr-thresh", type=float, default=0.99)
    ap.add_argument("--algo", default="lgbm", choices=["lgbm", "xgb"])
    
    args = ap.parse_args()

    # 1. Load
    print(f"[Filter] Loading {args.input}...")
    df = read_any(args.input)
    
    # Identify feature columns vs meta columns
    feat_cols = [c for c in df.columns if c.startswith(args.mordred_prefix)]
    meta_cols = [c for c in df.columns if c not in feat_cols]
    
    print(f"[Filter] Found {len(feat_cols)} candidate features (prefix '{args.mordred_prefix}').")
    
    if len(feat_cols) == 0:
        print("[Filter] No features found to filter. Saving copy.")
        df.to_parquet(args.output, index=False)
        return

    # 2. Drop Constant/NZV
    X = df[feat_cols]
    X_clean = drop_constant_and_nzv(X)
    print(f"[Filter] Dropped constant/NZV: {X.shape[1]} -> {X_clean.shape[1]} cols")

    # 3. Correlation Clustering
    clusters = correlation_clusters(X_clean, args.corr_thresh)
    print(f"[Filter] Found {len(clusters)} clusters (corr > {args.corr_thresh})")

    # 4. Selection by Importance
    y = None
    if args.target in df.columns:
        y = pd.to_numeric(df[args.target], errors="coerce").fillna(0.0)
    
    selected_cols = select_medoids(X_clean, y, clusters, args.algo)
    print(f"[Filter] Selected {len(selected_cols)} final features.")

    # 5. Reassemble & Save
    # Keep all meta columns + selected features
    final_df = pd.concat([df[meta_cols], X_clean[selected_cols]], axis=1)
    
    final_df.to_parquet(args.output, index=False)
    print(f"[Filter] Saved to {args.output}")

if __name__ == "__main__":
    main()