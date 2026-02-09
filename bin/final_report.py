#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Report: Comprehensive Visualization and Metrics Generation
================================================================

Generates visualization plots and metrics summary for model performance
evaluation. Supports both standard evaluation and cross-validation
consensus modes.

Plot Categories:
    plots_global/
        - Scatter plots (predicted vs actual)
        - Residual plots
        - Bias check plots
        - Error distribution histograms
    
    plots_physio/
        - Same plots for physiological temperature range (35-38°C)
    
    plots_classification/
        - ROC and PR curves
        - Confusion matrices

Classification Threshold:
    - logS > -4.0: Soluble (positive class, 1)
    - logS ≤ -4.0: Insoluble (negative class, 0)

Arguments:
    --level0   : Level-0 predictions parquet
    --blend    : Blend predictions parquet
    --stack    : Stack predictions parquet (optional)
    --metadata : Original test data with metadata
    --oof      : OOF predictions for train comparison
    --target   : Target column name (default: logS)
    --id-col   : ID column name (default: row_uid)
    --temp-col : Temperature column name (default: temp_C)
    --outdir   : Output directory
    --mode     : Report mode (standard or cv)

Usage:
    python final_report.py \\
        --level0 test_level0.parquet \\
        --blend test_blend.parquet \\
        --stack test_stack.parquet \\
        --metadata test_original.parquet \\
        --oof oof_predictions.parquet \\
        --target logS \\
        --mode standard \\
        --outdir .

Output:
    - plots_global/*.png: Global performance plots
    - plots_physio/*.png: Physiological range plots
    - plots_classification/*.png: Classification plots
    - metrics_summary.csv: Summary table
    - metrics_global.json: Detailed metrics JSON
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)

# Style configuration
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# Classification threshold
CLASSIFICATION_THRESHOLD = -4.0  # > -4 is Soluble (1), <= -4 is Insoluble (0)

# Column mappings (updated: tpsa → physics)
COL_MAP = {
    "y_xgb": "oof_xgb",
    "y_lgbm": "oof_lgbm",
    "y_cat": "oof_cat",
    "y_chemprop": "oof_gnn",
    "y_physics": "oof_physics",
    "y_pred_blend": "oof_blend",
    "y_pred_stack": "oof_stack"
}

MODEL_NAMES = {
    "y_xgb": "XGBoost",
    "y_lgbm": "LightGBM",
    "y_cat": "CatBoost",
    "y_chemprop": "Chemprop",
    "y_physics": "Physics",
    "y_pred_blend": "Blend",
    "y_pred_stack": "Stack"
}


# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

def calc_regression_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"rmse": None, "r2": None}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "r2": float(r2_score(y_true[mask], y_pred[mask]))
    }


def calc_classification_metrics(y_true_cont, y_pred_cont, threshold=CLASSIFICATION_THRESHOLD) -> dict:
    """Calculate binary classification metrics."""
    mask = np.isfinite(y_true_cont) & np.isfinite(y_pred_cont)
    if not mask.any():
        return {}
    
    # Binarize
    y_true = (y_true_cont[mask] > threshold).astype(int)
    y_pred = (y_pred_cont[mask] > threshold).astype(int)
    y_scores = y_pred_cont[mask]  # Continuous scores for AUC
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # AUC scores
    try:
        if len(np.unique(y_true)) >= 2:
            roc_auc_val = float(roc_auc_score(y_true, y_scores))
        else:
            roc_auc_val = 0.0
    except Exception:
        roc_auc_val = 0.0
    
    try:
        pr_auc_val = float(average_precision_score(y_true, y_scores))
    except Exception:
        pr_auc_val = 0.0
    
    return {
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val
    }


def normalize_target_col(df, target_name):
    """Ensure target column exists with correct name."""
    if target_name in df.columns:
        return df
    
    candidates = ["y_true", "y", "target", "response", "logS", "logs"]
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: target_name})
    
    # Check for y_true_* columns
    y_true_cols = [c for c in df.columns if c.startswith("y_true_")]
    if y_true_cols:
        return df.rename(columns={y_true_cols[0]: target_name})
    
    return df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_scatter_comparison(df_test, df_oof, target_col, test_col, oof_col, 
                           model_name, out_path, mode="standard"):
    """Generate scatter plot comparing predictions to actual values."""
    plt.figure(figsize=(8, 8))
    
    m_train = {"rmse": None, "r2": None}
    has_train = False
    
    # Plot Train (OOF) - only in standard mode
    if df_oof is not None and target_col in df_oof.columns:
        actual_col = oof_col if oof_col in df_oof.columns else test_col
        
        if actual_col and actual_col in df_oof.columns:
            y_tr = pd.to_numeric(df_oof[target_col], errors='coerce')
            y_pr = pd.to_numeric(df_oof[actual_col], errors='coerce')
            mask = np.isfinite(y_tr) & np.isfinite(y_pr)
            
            if mask.any():
                m_train = calc_regression_metrics(y_tr, y_pr)
                if mode != "cv" and not df_test.equals(df_oof):
                    sns.scatterplot(x=y_tr[mask], y=y_pr[mask], 
                                   color='royalblue', alpha=0.3, s=30, label="Train (OOF)")
                    has_train = True
    
    # Plot Test
    m_test = {"rmse": None, "r2": None}
    if test_col in df_test.columns:
        y_tr = pd.to_numeric(df_test[target_col], errors='coerce')
        y_pr = pd.to_numeric(df_test[test_col], errors='coerce')
        mask = np.isfinite(y_tr) & np.isfinite(y_pr)
        
        if mask.any():
            m_test = calc_regression_metrics(y_tr, y_pr)
            label_txt = "Consensus CV" if mode == "cv" else "Test"
            sns.scatterplot(x=y_tr[mask], y=y_pr[mask],
                           color='darkorange', alpha=0.6, s=40, edgecolor='w', label=label_txt)
    
    # Threshold lines
    plt.axvline(CLASSIFICATION_THRESHOLD, color='gray', linestyle=':', alpha=0.5,
                label=f'Threshold ({CLASSIFICATION_THRESHOLD})')
    plt.axhline(CLASSIFICATION_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    
    # Ideal line
    all_vals = []
    if has_train:
        all_vals.extend([df_oof[target_col].min(), df_oof[target_col].max()])
    all_vals.extend([df_test[target_col].min(), df_test[target_col].max()])
    all_vals = [v for v in all_vals if np.isfinite(v)]
    
    if all_vals:
        low, high = min(all_vals) - 0.5, max(all_vals) + 0.5
        plt.plot([low, high], [low, high], 'k--', lw=2, alpha=0.8, label='Ideal')
        plt.xlim(low, high)
        plt.ylim(low, high)
    
    plt.xlabel(f"Experimental {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    title_suffix = "(Cross-Validation Consensus)" if mode == "cv" else "Performance"
    plt.title(f"{model_name}: {title_suffix}")
    
    # Stats box (only in standard mode)
    if mode != "cv":
        tr_rmse = f"{m_train['rmse']:.3f}" if m_train['rmse'] else "N/A"
        tr_r2 = f"{m_train['r2']:.3f}" if m_train['r2'] else "N/A"
        te_rmse = f"{m_test['rmse']:.3f}" if m_test['rmse'] else "N/A"
        te_r2 = f"{m_test['r2']:.3f}" if m_test['r2'] else "N/A"
        
        stats_text = (f"Train (OOF)\nRMSE: {tr_rmse}\n$R^2$: {tr_r2}\n\n"
                     f"Test\nRMSE: {te_rmse}\n$R^2$: {te_r2}")
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return {
        "model": model_name,
        "test_rmse": m_test['rmse'],
        "test_r2": m_test['r2'],
        "train_rmse": m_train['rmse'],
        "train_r2": m_train['r2']
    }


def plot_residuals(df, target_col, pred_col, model_name, out_path):
    """Generate residual plot."""
    if pred_col not in df.columns or target_col not in df.columns:
        return
    
    y_true = pd.to_numeric(df[target_col], errors='coerce')
    y_pred = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if not mask.any():
        return
    
    residuals = y_true[mask] - y_pred[mask]
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_pred[mask], y=residuals, color='steelblue', alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual (True - Pred)")
    plt.title(f"{model_name}: Residuals vs Predicted")
    
    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color='steelblue', bins=30)
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.xlabel("Residual")
    plt.title(f"{model_name}: Residual Distribution")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bias_check(df, target_col, pred_col, model_name, out_path):
    """Generate bias check plot (residuals vs true values)."""
    if pred_col not in df.columns or target_col not in df.columns:
        return
    
    y_true = pd.to_numeric(df[target_col], errors='coerce')
    y_pred = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if not mask.any():
        return
    
    residuals = y_true[mask] - y_pred[mask]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true[mask], y=residuals, color='teal', alpha=0.5)
    plt.axhline(0, color='k', linestyle='--', lw=2)
    plt.xlabel("True Value")
    plt.ylabel("Residuals")
    plt.title(f"{model_name}: Bias Check")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_hist(df, target_col, pred_col, model_name, out_path):
    """Generate error histogram."""
    if pred_col not in df.columns or target_col not in df.columns:
        return
    
    y_true = pd.to_numeric(df[target_col], errors='coerce')
    y_pred = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if not mask.any():
        return
    
    error = y_true[mask] - y_pred[mask]
    
    plt.figure(figsize=(8, 5))
    sns.histplot(error, kde=True, color='crimson', bins=30)
    plt.axvline(0, color='k', linestyle='--')
    plt.title(f"{model_name}: Error Distribution")
    plt.xlabel("Error (True - Predicted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_classification_curves(df, target_col, pred_col, model_name, out_path, 
                               threshold=CLASSIFICATION_THRESHOLD):
    """Generate ROC and PR curves."""
    if pred_col not in df.columns or target_col not in df.columns:
        return
    
    y_true_cont = pd.to_numeric(df[target_col], errors='coerce')
    y_pred_cont = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true_cont) & np.isfinite(y_pred_cont)
    
    if not mask.any():
        return
    
    y_true = (y_true_cont[mask] > threshold).astype(int)
    y_scores = y_pred_cont[mask]
    
    if len(np.unique(y_true)) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc_val = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc_val:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{model_name}: ROC Curve')
    axes[0].legend(loc="lower right")
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_val = average_precision_score(y_true, y_scores)
    
    axes[1].plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {pr_auc_val:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{model_name}: PR Curve')
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(df, target_col, pred_col, model_name, out_path,
                          threshold=CLASSIFICATION_THRESHOLD):
    """Generate confusion matrix plot."""
    if pred_col not in df.columns or target_col not in df.columns:
        return
    
    y_true_cont = pd.to_numeric(df[target_col], errors='coerce')
    y_pred_cont = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true_cont) & np.isfinite(y_pred_cont)
    
    if not mask.any():
        return
    
    y_true = (y_true_cont[mask] > threshold).astype(int)
    y_pred = (y_pred_cont[mask] > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: Insoluble', 'Pred: Soluble'],
                yticklabels=['Real: Insoluble', 'Real: Soluble'])
    plt.title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for report generation."""
    
    ap = argparse.ArgumentParser(
        description="Generate comprehensive visualization plots and metrics."
    )
    ap.add_argument("--level0", required=True, help="Level-0 predictions parquet")
    ap.add_argument("--blend", required=True, help="Blend predictions parquet")
    ap.add_argument("--stack", default=None, help="Stack predictions parquet")
    ap.add_argument("--metadata", required=True, help="Original test data")
    ap.add_argument("--oof", required=True, help="OOF predictions")
    ap.add_argument("--target", default="logS", help="Target column")
    ap.add_argument("--id-col", default="row_uid", help="ID column")
    ap.add_argument("--temp-col", default="temp_C", help="Temperature column")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--bin-step", type=float, default=0.2, help="Bin step for histograms")
    ap.add_argument("--mode", default="standard", choices=["standard", "cv"],
                    help="Report mode")
    
    args = ap.parse_args()
    
    print(f"[Report] Loading data (mode: {args.mode})...")
    
    # Load data
    df_test = pd.read_parquet(args.level0)
    blend = pd.read_parquet(args.blend)
    meta = pd.read_parquet(args.metadata)
    df_oof = pd.read_parquet(args.oof)
    
    # Merge blend predictions
    if args.id_col in blend.columns:
        df_test = df_test.merge(blend[[args.id_col, "y_pred_blend"]], on=args.id_col, how="left")
    
    # Merge stack predictions
    if args.stack and Path(args.stack).exists():
        stk = pd.read_parquet(args.stack)
        if args.id_col in stk.columns:
            df_test = df_test.merge(stk[[args.id_col, "y_pred_stack"]], on=args.id_col, how="left")
    
    # Add metadata columns
    cols_to_add = [c for c in [args.target, args.temp_col] 
                   if c not in df_test.columns and c in meta.columns]
    if cols_to_add:
        df_test = df_test.merge(meta[[args.id_col] + cols_to_add], on=args.id_col, how="left")
    
    # Normalize target column
    df_test = normalize_target_col(df_test, args.target)
    df_oof = normalize_target_col(df_oof, args.target)
    
    # Create output directories
    out_global = Path(args.outdir) / "plots_global"
    out_physio = Path(args.outdir) / "plots_physio"
    out_class = Path(args.outdir) / "plots_classification"
    
    os.makedirs(out_global, exist_ok=True)
    os.makedirs(out_physio, exist_ok=True)
    os.makedirs(out_class, exist_ok=True)
    
    metrics_json = {}
    summary = []
    
    # -------------------------------------------------------------------------
    # Generate Global Plots
    # -------------------------------------------------------------------------
    print("[Report] Generating global plots...")
    
    for test_col, name in MODEL_NAMES.items():
        if test_col not in df_test.columns:
            continue
        
        oof_col = COL_MAP.get(test_col)
        # Handle legacy column names
        if oof_col == "oof_gnn" and "oof_chemprop" in df_oof.columns:
            oof_col = "oof_chemprop"
        
        # Regression plots
        res = plot_scatter_comparison(df_test, df_oof, args.target, test_col, oof_col,
                                     name, out_global / f"scatter_{test_col}.png", mode=args.mode)
        
        plot_residuals(df_test, args.target, test_col, name, out_global / f"residuals_{test_col}.png")
        plot_bias_check(df_test, args.target, test_col, name, out_global / f"bias_{test_col}.png")
        plot_error_hist(df_test, args.target, test_col, name, out_global / f"error_hist_{test_col}.png")
        
        # Classification metrics and plots
        y_true_vals = pd.to_numeric(df_test[args.target], errors='coerce')
        y_pred_vals = pd.to_numeric(df_test[test_col], errors='coerce')
        
        class_metrics = calc_classification_metrics(y_true_vals, y_pred_vals)
        
        plot_classification_curves(df_test, args.target, test_col, name,
                                  out_class / f"roc_pr_{test_col}.png")
        plot_confusion_matrix(df_test, args.target, test_col, name,
                             out_class / f"conf_matrix_{test_col}.png")
        
        # Combine results
        res.update(class_metrics)
        res["scope"] = "Global"
        summary.append(res)
        metrics_json[test_col] = res
    
    # -------------------------------------------------------------------------
    # Generate Physiological Range Plots
    # -------------------------------------------------------------------------
    print("[Report] Generating physiological range plots...")
    
    if args.temp_col in df_test.columns:
        t_test = pd.to_numeric(df_test[args.temp_col], errors='coerce')
        mask_phys = t_test.between(35, 38)
        df_test_phys = df_test[mask_phys]
        
        if not df_test_phys.empty:
            metrics_json["physio_range"] = {}
            
            for test_col, name in MODEL_NAMES.items():
                if test_col not in df_test_phys.columns:
                    continue
                
                oof_col = COL_MAP.get(test_col)
                if oof_col == "oof_gnn" and "oof_chemprop" in df_oof.columns:
                    oof_col = "oof_chemprop"
                
                res = plot_scatter_comparison(df_test_phys, df_oof, args.target, test_col, oof_col,
                                             name, out_physio / f"scatter_{test_col}.png", mode=args.mode)
                
                # Classification metrics for physio range
                y_true_p = pd.to_numeric(df_test_phys[args.target], errors='coerce')
                y_pred_p = pd.to_numeric(df_test_phys[test_col], errors='coerce')
                class_m_p = calc_classification_metrics(y_true_p, y_pred_p)
                
                res.update(class_m_p)
                res["scope"] = "Physio"
                summary.append(res)
                metrics_json["physio_range"][test_col] = res
                
                plot_residuals(df_test_phys, args.target, test_col, name,
                              out_physio / f"residuals_{test_col}.png")
                plot_bias_check(df_test_phys, args.target, test_col, name,
                               out_physio / f"bias_{test_col}.png")
                plot_error_hist(df_test_phys, args.target, test_col, name,
                               out_physio / f"error_hist_{test_col}.png")
    
    # -------------------------------------------------------------------------
    # Save Summary
    # -------------------------------------------------------------------------
    pd.DataFrame(summary).to_csv(Path(args.outdir) / "metrics_summary.csv", index=False)
    
    with open(Path(args.outdir) / "metrics_global.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print("[Report] Done. Classification metrics included.")


if __name__ == "__main__":
    main()
