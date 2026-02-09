#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidate Parameters: Hyperparameter Aggregation Across Folds
================================================================

Aggregates hyperparameters from multiple fold-specific JSON files into a
single consolidated parameter set for final model training.

Aggregation Strategy:
    - Numeric (float): Mean value across folds
    - Numeric (int): Rounded mean value across folds
    - Categorical/Boolean: Mode (most frequent value)

Arguments:
    input_files : List of JSON files or directories containing JSON files
    --output    : Output filename (default: best_params_consolidated.json)

Usage:
    python consolidate_params.py fold_0/best_params.json fold_1/best_params.json \\
        --output best_params_consolidated.json
    
    # Or with directories:
    python consolidate_params.py fold_0/ fold_1/ fold_2/ \\
        --output best_params_consolidated.json

Output:
    JSON file with aggregated parameters

Notes:
    - Directories are searched for .json files
    - Missing parameters in some folds are ignored
    - Empty or invalid JSON files are skipped with warnings
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def find_json_in_directory(directory: str) -> str:
    """
    Find JSON file in directory.
    
    Args:
        directory: Path to directory
    
    Returns:
        Path to first JSON file found, or None
    """
    candidates = glob.glob(os.path.join(directory, "*.json"))
    if not candidates:
        return None
    return candidates[0]


def aggregate_parameters(all_params: dict) -> dict:
    """
    Aggregate parameters using appropriate strategy per type.
    
    Args:
        all_params: Dictionary mapping param name to list of values
    
    Returns:
        Dictionary with aggregated parameter values
    """
    final_params = {}
    
    for key, values in all_params.items():
        if not values:
            continue
        
        first_val = values[0]
        
        # Numeric (non-boolean)
        if isinstance(first_val, (int, float)) and not isinstance(first_val, bool):
            mean_val = np.mean(values)
            
            # If all values were integers, round to integer
            if all(isinstance(x, int) for x in values):
                final_params[key] = int(round(mean_val))
            else:
                final_params[key] = float(mean_val)
        
        # Categorical or boolean
        else:
            counts = Counter(values)
            final_params[key] = counts.most_common(1)[0][0]
    
    return final_params


def main():
    """Main entry point for parameter consolidation."""
    
    ap = argparse.ArgumentParser(
        description="Consolidate hyperparameters from multiple folds."
    )
    ap.add_argument('input_files', nargs='+',
                    help='List of JSON files or directories')
    ap.add_argument('--output', default='best_params_consolidated.json',
                    help='Output filename')
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Collect Parameters from All Inputs
    # -------------------------------------------------------------------------
    all_params = {}
    files_processed = 0
    
    print(f"[Consolidate] Processing {len(args.input_files)} inputs...")
    
    for input_path in args.input_files:
        target_file = input_path
        
        # If directory, find JSON inside
        if os.path.isdir(input_path):
            target_file = find_json_in_directory(input_path)
            if not target_file:
                print(f"[WARN] No .json found in directory: {input_path}")
                continue
            print(f"         Directory detected. Using: {target_file}")
        
        # Read JSON file
        try:
            with open(target_file, 'r') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(value)
            
            files_processed += 1
            
        except Exception as e:
            print(f"[WARN] Error reading {target_file}: {e}")
    
    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------
    if not all_params:
        print("[ERROR] No parameters loaded. Check input files.")
        sys.exit(1)
    
    print(f"[Consolidate] Loaded parameters from {files_processed} files")
    print(f"[Consolidate] Parameters found: {list(all_params.keys())}")
    
    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    final_params = aggregate_parameters(all_params)
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    with open(args.output, 'w') as f:
        json.dump(final_params, f, indent=4)
    
    print(f"\n[Consolidate] Saved to: {args.output}")
    print(f"[Consolidate] Final parameters:")
    for key, value in final_params.items():
        print(f"         {key}: {value}")


if __name__ == "__main__":
    main()
