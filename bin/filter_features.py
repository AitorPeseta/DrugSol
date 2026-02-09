#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter Features: Correlation-Based Feature Selection with Importance Ranking
=============================================================================

Advanced feature selection pipeline that reduces dimensionality while preserving
predictive information. Particularly useful for Mordred descriptors which contain
many highly correlated features.

Selection Pipeline:
    1. Remove constant and near-zero variance (NZV) columns
    2. Build correlation graph and find connected components (clusters)
    3. Select one representative per cluster using feature importance (gain)

This approach typically reduces ~1800 Mordred descriptors to ~200-400 features
while maintaining model performance.

Arguments:
    --input, -i      : Input Parquet/CSV file with features
    --output, -o     : Output Parquet file path
    --target         : Target column for importance calculation (default: logS)
    --mordred-prefix : Prefix identifying feature columns to filter (default: mordred__)
    --corr-thresh    : Correlation threshold for clustering (default: 0.99)
    --algo           : Algorithm for importance: lgbm or xgb (default: lgbm)

Usage:
    python filter_features.py \\
        --input train_mordred_featured.parquet \\
        --output train_features_mordred_filtered.parquet \\
        --target logS \\
        --mordred-prefix "mordred__" \\
        --corr-thresh 0.99 \\
        --algo lgbm

Output:
    Parquet file containing:
    - All non-feature columns (metadata) preserved
    - Filtered subset of feature columns (one per correlation cluster)

Notes:
    - Features not matching the prefix are preserved unchanged
    - Missing target column triggers variance-based fallback selection
    - Correlation clustering uses graph-based connected components
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd

# Optional GBM imports for importance calculation
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


# ============================================================================
# I/O UTILITIES
# ============================================================================

def read_any(path: str) -> pd.DataFrame:
    """Read input file as DataFrame (supports Parquet and CSV)."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ============================================================================
# FEATURE FILTERING FUNCTIONS
# ============================================================================

def drop_constant_and_nzv(df: pd.DataFrame, nzv_thresh: float = 0.01) -> pd.DataFrame:
    """
    Remove constant and near-zero variance columns.
    
    A column is considered NZV if the most frequent value appears in
    more than (1 - nzv_thresh) proportion of rows.
    
    Args:
        df: DataFrame with numeric columns
        nzv_thresh: Threshold for near-zero variance (default: 0.01)
    
    Returns:
        DataFrame with constant and NZV columns removed
    """
    numeric_df = df.select_dtypes(include=[np.number])
    keep_cols = []
    
    for col in numeric_df.columns:
        series = numeric_df[col]
        
        # Skip constant columns
        if series.nunique(dropna=True) <= 1:
            continue
        
        # Skip NZV columns (single value dominates)
        vc = series.value_counts(normalize=True, dropna=True)
        if not vc.empty and vc.iloc[0] >= (1.0 - nzv_thresh):
            continue
        
        keep_cols.append(col)
    
    return numeric_df[keep_cols]


def correlation_clusters(df: pd.DataFrame, corr_thresh: float) -> List[List[str]]:
    """
    Group features into clusters based on correlation.
    
    Builds a graph where edges connect features with |correlation| > threshold,
    then finds connected components (clusters).
    
    Args:
        df: DataFrame with numeric features
        corr_thresh: Absolute correlation threshold
    
    Returns:
        List of clusters, where each cluster is a list of column names
    """
    if df.shape[1] <= 1:
        return [df.columns.tolist()]
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs().fillna(0.0)
    cols = corr_matrix.columns.tolist()
    
    # Build adjacency graph
    adj: Dict[str, Set[str]] = {c: set() for c in cols}
    for i, c1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            c2 = cols[j]
            if corr_matrix.iloc[i, j] >= corr_thresh:
                adj[c1].add(c2)
                adj[c2].add(c1)
    
    # Find connected components using DFS
    visited = set()
    clusters = []
    
    for node in cols:
        if node in visited:
            continue
        
        # DFS to find component
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


def get_feature_importance_gain(
    X: pd.DataFrame,
    y: pd.Series,
    algo: str = "lgbm"
) -> pd.Series:
    """
    Calculate feature importance using gradient boosting gain.
    
    Trains a quick GBM model and extracts gain-based importance scores.
    Falls back to variance if no valid target or GBM unavailable.
    
    Args:
        X: Feature DataFrame
        y: Target Series (can be None)
        algo: Algorithm to use ("lgbm" or "xgb")
    
    Returns:
        Series of importance scores indexed by feature name
    """
    # Fallback if no valid target
    if y is None or y.isna().all():
        var = X.var()
        return var / (var.sum() or 1.0)
    
    # LightGBM
    if algo == "lgbm" and _HAS_LGBM:
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "seed": 42
        }
        model = lgb.train(params, dtrain, num_boost_round=100)
        gain = model.feature_importance(importance_type="gain")
        return pd.Series(gain, index=X.columns)
    
    # XGBoost
    if algo == "xgb" and _HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        model.fit(X, y)
        gain = model.feature_importances_
        return pd.Series(gain, index=X.columns)
    
    # Final fallback to variance
    var = X.var()
    return var / (var.sum() or 1.0)


def select_medoids(
    X: pd.DataFrame,
    y: pd.Series,
    clusters: List[List[str]],
    algo: str = "lgbm"
) -> List[str]:
    """
    Select the best feature from each correlation cluster.
    
    Uses feature importance (gain) to pick the most predictive
    representative from each cluster of correlated features.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        clusters: List of feature clusters
        algo: Algorithm for importance calculation
    
    Returns:
        List of selected feature names (one per cluster)
    """
    selected_features = []
    
    # Calculate global importance once
    importances = get_feature_importance_gain(X, y, algo)
    
    for cluster in clusters:
        if len(cluster) == 1:
            selected_features.append(cluster[0])
            continue
        
        # Pick feature with maximum importance in this cluster
        cluster_importance = importances.reindex(cluster).fillna(0.0)
        best_feat = cluster_importance.idxmax()
        selected_features.append(best_feat)
    
    return selected_features


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for feature filtering script."""
    
    ap = argparse.ArgumentParser(
        description="Filter features by correlation clustering and importance."
    )
    ap.add_argument("-i", "--input", required=True,
                    help="Input Parquet/CSV file")
    ap.add_argument("-o", "--output", required=True,
                    help="Output Parquet file")
    ap.add_argument("--target", default="logS",
                    help="Target column for importance (default: logS)")
    ap.add_argument("--mordred-prefix", default="mordred__",
                    help="Prefix of feature columns to filter")
    ap.add_argument("--corr-thresh", type=float, default=0.99,
                    help="Correlation threshold (default: 0.99)")
    ap.add_argument("--algo", default="lgbm", choices=["lgbm", "xgb"],
                    help="Importance algorithm (default: lgbm)")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"[Filter] Loading {args.input}...")
    df = read_any(args.input)
    print(f"[Filter] Loaded {len(df):,} rows, {len(df.columns):,} columns")
    
    # -------------------------------------------------------------------------
    # Identify Feature vs Meta Columns
    # -------------------------------------------------------------------------
    feat_cols = [c for c in df.columns if c.startswith(args.mordred_prefix)]
    meta_cols = [c for c in df.columns if c not in feat_cols]
    
    print(f"[Filter] Found {len(feat_cols)} feature columns (prefix '{args.mordred_prefix}')")
    print(f"[Filter] Preserving {len(meta_cols)} metadata columns")
    
    if len(feat_cols) == 0:
        print("[Filter] No features found to filter. Saving copy.")
        df.to_parquet(args.output, index=False)
        return
    
    # -------------------------------------------------------------------------
    # Step 1: Remove Constant/NZV
    # -------------------------------------------------------------------------
    X = df[feat_cols]
    X_clean = drop_constant_and_nzv(X)
    n_dropped_nzv = X.shape[1] - X_clean.shape[1]
    print(f"[Filter] Step 1: Removed {n_dropped_nzv} constant/NZV columns")
    print(f"         Remaining: {X_clean.shape[1]} columns")
    
    # -------------------------------------------------------------------------
    # Step 2: Correlation Clustering
    # -------------------------------------------------------------------------
    clusters = correlation_clusters(X_clean, args.corr_thresh)
    n_singletons = sum(1 for c in clusters if len(c) == 1)
    n_groups = len(clusters) - n_singletons
    print(f"[Filter] Step 2: Found {len(clusters)} clusters (corr > {args.corr_thresh})")
    print(f"         Singletons: {n_singletons}, Groups: {n_groups}")
    
    # -------------------------------------------------------------------------
    # Step 3: Importance-Based Selection
    # -------------------------------------------------------------------------
    y = None
    if args.target in df.columns:
        y = pd.to_numeric(df[args.target], errors="coerce").fillna(0.0)
    else:
        print(f"[WARN] Target column '{args.target}' not found. Using variance fallback.")
    
    selected_cols = select_medoids(X_clean, y, clusters, args.algo)
    print(f"[Filter] Step 3: Selected {len(selected_cols)} representative features")
    
    # -------------------------------------------------------------------------
    # Assemble and Save
    # -------------------------------------------------------------------------
    final_df = pd.concat([df[meta_cols], X_clean[selected_cols]], axis=1)
    
    final_df.to_parquet(args.output, index=False)
    
    # Summary
    reduction = (1 - len(selected_cols) / len(feat_cols)) * 100
    print(f"\n[Filter] Summary:")
    print(f"         Input features: {len(feat_cols)}")
    print(f"         Output features: {len(selected_cols)}")
    print(f"         Reduction: {reduction:.1f}%")
    print(f"         Saved to: {args.output}")


if __name__ == "__main__":
    main()
