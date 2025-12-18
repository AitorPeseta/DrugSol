#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_report.py
---------------
Generates plots and metrics.
Features: 
- Hides metrics text box if --mode cv is used.
- Adds Classification Metrics (Acc, Spec, Prec, Rec, F1, MCC, ROC, PR).
- Threshold for Solubility: -4.0 logS
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from pathlib import Path

# Estilo profesional
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# CONSTANTES
CLASSIFICATION_THRESHOLD = -4.0  # > -4 es Soluble (1), <= -4 es Insoluble (0)

COL_MAP = {
    "y_xgb": "oof_xgb", "y_lgbm": "oof_lgbm", "y_chemprop": "oof_gnn",
    "y_tpsa": "oof_tpsa", "y_pred_blend": "oof_blend", "y_pred_stack": "oof_stack"
}

MODEL_NAMES = {
    "y_xgb": "XGBoost", "y_lgbm": "LightGBM", "y_chemprop": "Chemprop",
    "y_tpsa": "Baseline", "y_pred_blend": "Blend", "y_pred_stack": "Stack"
}

def calc_regression_metrics(y_true, y_pred):
    """Métricas de Regresión (Continuas)"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return {"rmse": None, "r2": None}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "r2": float(r2_score(y_true[mask], y_pred[mask]))
    }

def calc_classification_metrics(y_true_cont, y_pred_cont, threshold=CLASSIFICATION_THRESHOLD):
    """Métricas de Clasificación (Binarias)"""
    # 1. Binarizar
    mask = np.isfinite(y_true_cont) & np.isfinite(y_pred_cont)
    if not mask.any(): return {}
    
    y_true = (y_true_cont[mask] > threshold).astype(int)
    y_pred = (y_pred_cont[mask] > threshold).astype(int)
    y_scores = y_pred_cont[mask] # Usamos el valor continuo como score para AUC
    
    # 2. Matriz de Confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 3. Cálculos
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 4. AUCs (Requieren scores continuos, no binarios)
    try:
        roc_auc = float(from_sklearn_roc_auc(y_true, y_scores))
    except:
        roc_auc = 0.0
        
    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except:
        pr_auc = 0.0

    return {
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

# Helper para ROC AUC seguro
def from_sklearn_roc_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2: return 0.0
    return roc_auc_score(y_true, y_score)

def normalize_target_col(df, target_name):
    if target_name in df.columns: return df
    candidates = ["y_true", "y", "target", "response", "logS", "logs"]
    for c in candidates:
        if c in df.columns: return df.rename(columns={c: target_name})
    y_true_cols = [c for c in df.columns if c.startswith("y_true_")]
    if y_true_cols: return df.rename(columns={y_true_cols[0]: target_name})
    return df

def plot_scatter_comparison(df_test, df_oof, target_col, test_col, oof_col, model_name, out_path, mode="standard"):
    plt.figure(figsize=(8, 8))
    
    m_train = {"rmse": None, "r2": None}
    has_train = False

    # Plot Train (Solo si no es CV)
    if df_oof is not None and target_col in df_oof.columns:
        actual_col = None
        if oof_col in df_oof.columns: actual_col = oof_col
        elif test_col in df_oof.columns: actual_col = test_col
        
        if actual_col:
            y_tr = pd.to_numeric(df_oof[target_col], errors='coerce')
            y_pr = pd.to_numeric(df_oof[actual_col], errors='coerce')
            mask = np.isfinite(y_tr) & np.isfinite(y_pr)
            if mask.any():
                m_train = calc_regression_metrics(y_tr, y_pr)
                if mode != "cv" and not df_test.equals(df_oof):
                     sns.scatterplot(x=y_tr[mask], y=y_pr[mask], color='royalblue', alpha=0.3, s=30, label="Train (OOF)")
                     has_train = True

    # Plot Test
    m_test = {"rmse": None, "r2": None}
    if test_col in df_test.columns:
        y_tr = pd.to_numeric(df_test[target_col], errors='coerce')
        y_pr = pd.to_numeric(df_test[test_col], errors='coerce')
        mask = np.isfinite(y_tr) & np.isfinite(y_pr)
        if mask.any():
            m_test = calc_regression_metrics(y_tr, y_pr)
            label_txt = "Consenso CV" if mode == "cv" else "Test"
            sns.scatterplot(x=y_tr[mask], y=y_pr[mask], color='darkorange', alpha=0.6, s=40, edgecolor='w', label=label_txt)

    # Threshold Line
    plt.axvline(CLASSIFICATION_THRESHOLD, color='gray', linestyle=':', alpha=0.5, label=f'Threshold ({CLASSIFICATION_THRESHOLD})')
    plt.axhline(CLASSIFICATION_THRESHOLD, color='gray', linestyle=':', alpha=0.5)

    # Ideal Line
    all_vals = []
    if has_train: all_vals.extend([df_oof[target_col].min(), df_oof[target_col].max()])
    all_vals.extend([df_test[target_col].min(), df_test[target_col].max()])
    all_vals = [v for v in all_vals if np.isfinite(v)]
    
    if all_vals:
        low, high = min(all_vals)-0.5, max(all_vals)+0.5
        plt.plot([low, high], [low, high], 'k--', lw=2, alpha=0.8, label='Ideal')
        plt.xlim(low, high); plt.ylim(low, high)

    plt.xlabel(f"Experimental {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    title_suffix = "(Cross-Validation Consensus)" if mode == "cv" else "Performance"
    plt.title(f"{model_name}: {title_suffix}")
    
    if mode != "cv":
        tr_rmse = f"{m_train['rmse']:.3f}" if m_train['rmse'] else "N/A"
        tr_r2   = f"{m_train['r2']:.3f}" if m_train['r2'] else "N/A"
        te_rmse = f"{m_test['rmse']:.3f}" if m_test['rmse'] else "N/A"
        te_r2   = f"{m_test['r2']:.3f}" if m_test['r2'] else "N/A"
        
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
        "test_rmse": m_test['rmse'], "test_r2": m_test['r2'], 
        "train_rmse": m_train['rmse'], "train_r2": m_train['r2']
    }

def plot_classification_curves(df, target_col, pred_col, model_name, out_path, threshold=CLASSIFICATION_THRESHOLD):
    """Genera curvas ROC y PR"""
    if pred_col not in df.columns or target_col not in df.columns: return
    
    # Preparar datos
    y_true_cont = pd.to_numeric(df[target_col], errors='coerce')
    y_scores = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true_cont) & np.isfinite(y_scores)
    
    if not mask.any(): return
    y_true_cont = y_true_cont[mask]
    y_scores = y_scores[mask]
    
    # Binarizar Realidad
    y_true_bin = (y_true_cont > threshold).astype(int)
    if len(np.unique(y_true_bin)) < 2: return # Necesitamos ambas clases para plotear
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Recall)')
    ax1.set_title(f'{model_name}: ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
    pr_auc = average_precision_score(y_true_bin, y_scores)
    
    ax2.plot(recall, precision, color='purple', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{model_name}: Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals(df, target_col, pred_col, model_name, out_path):
    """
    Gráfico de residuos: (y_true - y_pred) vs y_pred
    CORREGIDO: Ahora acepta explícitamente target y pred column.
    """
    if pred_col not in df.columns or target_col not in df.columns:
        return

    y_true = pd.to_numeric(df[target_col], errors='coerce')
    y_pred = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if not mask.any(): return

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 6))
    
    # Colores genéricos
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color='royalblue')
    plt.axhline(0, color='black', linestyle='--')
    
    plt.xlabel(f"Predicted {target_col}")
    plt.ylabel("Residual (True - Pred)")
    plt.title(f"{model_name}: Residual Analysis")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_bias_check(df, target_col, pred_col, model_name, out_path):
    if pred_col not in df.columns or target_col not in df.columns: return
    y_true = pd.to_numeric(df[target_col], errors='coerce')
    y_pred = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return
    res = y_true[mask] - y_pred[mask]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true[mask], y=res, color='teal', alpha=0.5)
    plt.axhline(0, color='k', linestyle='--', lw=2)
    plt.xlabel(f"True Value")
    plt.ylabel("Residuals")
    plt.title(f"{model_name}: Bias")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_error_hist(df, target_col, pred_col, model_name, out_path):
    if pred_col not in df.columns or target_col not in df.columns: return
    y_true = pd.to_numeric(df[target_col], errors='coerce')
    y_pred = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return
    error = y_true[mask] - y_pred[mask]
    plt.figure(figsize=(8, 5))
    sns.histplot(error, kde=True, color='crimson', bins=30)
    plt.axvline(0, color='k', linestyle='--')
    plt.title(f"{model_name}: Error Dist")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_confusion_matrix(df, target_col, pred_col, model_name, out_path, threshold=-4.0):
    """Genera y guarda la Matriz de Confusión visual"""
    if pred_col not in df.columns or target_col not in df.columns: return
    
    # Preparar datos
    y_true_cont = pd.to_numeric(df[target_col], errors='coerce')
    y_pred_cont = pd.to_numeric(df[pred_col], errors='coerce')
    mask = np.isfinite(y_true_cont) & np.isfinite(y_pred_cont)
    
    if not mask.any(): return
    
    # Binarizar (> -4 es Soluble/1, <= -4 es Insoluble/0)
    y_true = (y_true_cont[mask] > threshold).astype(int)
    y_pred = (y_pred_cont[mask] > threshold).astype(int)
    
    # Calcular matriz
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: Insoluble (0)', 'Pred: Soluble (1)'],
                yticklabels=['Real: Insoluble (0)', 'Real: Soluble (1)'])
    
    plt.title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level0", required=True)
    ap.add_argument("--blend", required=True)
    ap.add_argument("--stack", default=None)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--target", default="logS")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--temp-col", default="temp_C")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--bin-step", type=float, default=0.2)
    ap.add_argument("--mode", default="standard", choices=["standard", "cv"], 
                    help="Modo de reporte: standard o cv")
    
    args = ap.parse_args()

    print(f"Loading Data (Mode: {args.mode})...")
    df_test = pd.read_parquet(args.level0)
    blend = pd.read_parquet(args.blend)
    meta = pd.read_parquet(args.metadata)
    df_oof = pd.read_parquet(args.oof)
    
    # Merges
    if args.id_col in blend.columns:
        df_test = df_test.merge(blend[[args.id_col, "y_pred_blend"]], on=args.id_col, how="left")
    
    if args.stack and Path(args.stack).exists():
        stk = pd.read_parquet(args.stack)
        if args.id_col in stk.columns:
            df_test = df_test.merge(stk[[args.id_col, "y_pred_stack"]], on=args.id_col, how="left")
    
    cols_to_add = [c for c in [args.target, args.temp_col] if c not in df_test.columns and c in meta.columns]
    if cols_to_add: 
        df_test = df_test.merge(meta[[args.id_col] + cols_to_add], on=args.id_col, how="left")
    
    df_test = normalize_target_col(df_test, args.target)
    df_oof = normalize_target_col(df_oof, args.target)

    # Carpetas de salida
    out_global = Path(args.outdir) / "plots_global"
    out_physio = Path(args.outdir) / "plots_physio"
    out_class = Path(args.outdir) / "plots_classification" # NUEVA CARPETA
    
    os.makedirs(out_global, exist_ok=True)
    os.makedirs(out_physio, exist_ok=True)
    os.makedirs(out_class, exist_ok=True)
    
    metrics_json = {}
    summary = []

    print("Generating GLOBAL Plots & Classification Metrics...")
    for test_col, name in MODEL_NAMES.items():
        if test_col in df_test.columns:
            oof_col = COL_MAP.get(test_col)
            if oof_col == "oof_gnn" and "oof_chemprop" in df_oof.columns: oof_col = "oof_chemprop"
            
            # 1. Regresión y Scatters
            res = plot_scatter_comparison(df_test, df_oof, args.target, test_col, oof_col, name, out_global / f"scatter_{test_col}.png", mode=args.mode)
            
            # 2. Plots de Residuos, Bias, Hist
            # AQUI ESTABA EL ERROR: Ahora la función acepta los 5 argumentos correctamente
            plot_residuals(df_test, args.target, test_col, name, out_global / f"residuals_{test_col}.png")
            plot_bias_check(df_test, args.target, test_col, name, out_global / f"bias_{test_col}.png")
            plot_error_hist(df_test, args.target, test_col, name, out_global / f"error_hist_{test_col}.png")
            
            # 3. NUEVO: Métricas de Clasificación
            y_true_vals = pd.to_numeric(df_test[args.target], errors='coerce')
            y_pred_vals = pd.to_numeric(df_test[test_col], errors='coerce')
            
            class_metrics = calc_classification_metrics(y_true_vals, y_pred_vals, threshold=CLASSIFICATION_THRESHOLD)
            
            # 4. NUEVO: Curvas ROC y PR
            plot_classification_curves(df_test, args.target, test_col, name, out_class / f"roc_pr_{test_col}.png", threshold=CLASSIFICATION_THRESHOLD)

            # 5. NUEVO: Matriz de Confusión
            plot_confusion_matrix(df_test, args.target, test_col, name, out_class / f"conf_matrix_{test_col}.png", threshold=CLASSIFICATION_THRESHOLD)
            
            # Unir todo en el reporte
            res.update(class_metrics)
            res["scope"] = "Global"
            summary.append(res)
            metrics_json[test_col] = res

    print("Generating PHYSIO Plots...")
    if args.temp_col in df_test.columns:
        t_test = pd.to_numeric(df_test[args.temp_col], errors='coerce')
        mask_phys = t_test.between(35, 38)
        df_test_phys = df_test[mask_phys]
        
        if not df_test_phys.empty:
            metrics_json["physio_range"] = {}
            for test_col, name in MODEL_NAMES.items():
                if test_col in df_test_phys.columns:
                    oof_col = COL_MAP.get(test_col)
                    if oof_col == "oof_gnn" and "oof_chemprop" in df_oof.columns: oof_col = "oof_chemprop"

                    res = plot_scatter_comparison(df_test_phys, df_oof, args.target, test_col, oof_col, name, out_physio / f"scatter_{test_col}.png", mode=args.mode)
                    
                    # Calcular métricas clasificación también para Physio
                    y_true_p = pd.to_numeric(df_test_phys[args.target], errors='coerce')
                    y_pred_p = pd.to_numeric(df_test_phys[test_col], errors='coerce')
                    class_m_p = calc_classification_metrics(y_true_p, y_pred_p)
                    
                    res.update(class_m_p)
                    res["scope"] = "Physio"
                    summary.append(res)
                    metrics_json["physio_range"][test_col] = res
                    
                    # CORREGIDO AQUÍ TAMBIÉN
                    plot_residuals(df_test_phys, args.target, test_col, name, out_physio / f"residuals_{test_col}.png")
                    plot_bias_check(df_test_phys, args.target, test_col, name, out_physio / f"bias_{test_col}.png")
                    plot_error_hist(df_test_phys, args.target, test_col, name, out_physio / f"error_hist_{test_col}.png")

    # Guardar resumen final
    pd.DataFrame(summary).to_csv(Path(args.outdir) / "metrics_summary.csv", index=False)
    with open(Path(args.outdir) / "metrics_global.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
        
    print("Done. Classification metrics included.")

if __name__ == "__main__":
    main()