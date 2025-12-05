#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_validation.py (Fixed: JSON Output)
----------------------------------------
1. Analiza Métricas (Reporte CSV).
2. Fusiona Datos (Promedio por molécula).
3. Genera archivos Parquet para gráficas.
4. Genera JSON de métricas de consenso (Vital para Nextflow).
"""

import argparse
import json
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

def get_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return {"rmse": np.nan, "r2": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "r2": float(r2_score(y_true[mask], y_pred[mask]))
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Archivos metrics_*.json")
    ap.add_argument("--level0-files", nargs="+", required=True)
    ap.add_argument("--blend-files", nargs="+", required=True)
    ap.add_argument("--stack-files", nargs="+", default=[]) 
    ap.add_argument("--out-csv", default="montecarlo_summary.csv")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    MODELS = ["y_xgb", "y_lgbm", "y_chemprop", "y_tpsa", "y_pred_blend", "y_pred_stack"]
    data_global = {m: {"test_rmse": [], "test_r2": [], "train_rmse": [], "train_r2": []} for m in MODELS}
    
    valid_files = 0
    for fpath in args.inputs:
        try:
            with open(fpath, 'r') as f:
                d = json.load(f)
            for m in MODELS:
                if m in d:
                    # Test
                    if "test_rmse" in d[m]:
                        data_global[m]["test_rmse"].append(d[m]["test_rmse"])
                        data_global[m]["test_r2"].append(d[m]["test_r2"])
                    elif "rmse" in d[m]:
                         data_global[m]["test_rmse"].append(d[m]["rmse"])
                         data_global[m]["test_r2"].append(d[m]["r2"])
                    # Train
                    if "train_rmse" in d[m]:
                        data_global[m]["train_rmse"].append(d[m]["train_rmse"])
                        data_global[m]["train_r2"].append(d[m]["train_r2"])
            valid_files += 1
        except: pass

    rows = []
    for m in MODELS:
        te_rmse = np.array(data_global[m]["test_rmse"])
        te_r2   = np.array(data_global[m]["test_r2"])
        tr_rmse = np.array(data_global[m]["train_rmse"])
        tr_r2   = np.array(data_global[m]["train_r2"])
        
        if len(te_rmse) == 0: continue
        
        rows.append({
            "Model": m,
            "Test_RMSE": np.mean(te_rmse), "Test_RMSE_Std": np.std(te_rmse),
            "Test_R2": np.mean(te_r2),     "Test_R2_Std": np.std(te_r2),
            "Train_RMSE": np.mean(tr_rmse) if len(tr_rmse)>0 else None,
            "Train_R2": np.mean(tr_r2) if len(tr_r2)>0 else None
        })

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values("Test_RMSE")
        df_res.to_csv(outdir / args.out_csv, index=False)
        
        # Decidir Ganador
        winner = df_res.iloc[0]
        best_model = winner['Model'].replace("y_pred_", "").replace("y_", "")
        if best_model not in ["blend", "stack"]: best_model = "blend"
    else:
        best_model = "blend"

    with open("best_strategy.txt", "w") as f:
        f.write(best_model)

    print(f"[CV] Fusionando datos de {len(args.level0_files)} splits...")
    
    l0_files = sorted(args.level0_files)
    bl_files = sorted(args.blend_files)
    st_files = sorted(args.stack_files) if args.stack_files else [None]*len(l0_files)
    
    dfs = []
    for f0, fb, fs in zip(l0_files, bl_files, st_files):
        try:
            d0 = pd.read_parquet(f0)
            db = pd.read_parquet(fb)
            # Merge seguro por ID
            m = d0.merge(db[["row_uid", "y_pred_blend"]], on="row_uid", how="left")
            if fs:
                ds = pd.read_parquet(fs)
                m = m.merge(ds[["row_uid", "y_pred_stack"]], on="row_uid", how="left")
            dfs.append(m)
        except: pass
        
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        
        # --- PROMEDIO POR MOLÉCULA ---
        # Aquí es donde calculamos el valor medio de cada punto
        num_cols = big.select_dtypes(include=[np.number]).columns.tolist()
        if "fold" in num_cols: num_cols.remove("fold")
        
        df_mean = big.groupby("row_uid")[num_cols].mean().reset_index()
        print(f"[CV] Moléculas únicas (Consenso): {len(df_mean)}")
        
        # Guardar Parquets para el Reporte
        c_l0 = [c for c in df_mean.columns if c.startswith("y_") and "blend" not in c and "stack" not in c]
        # Asegurar metadatos
        cols_out = ["row_uid"]
        if "logS" in df_mean.columns: cols_out.append("logS")
        if "temp_C" in df_mean.columns: cols_out.append("temp_C")
        cols_out += c_l0
        
        df_mean[cols_out].to_parquet(outdir / "test_level0.parquet", index=False)
        
        if "y_pred_blend" in df_mean.columns:
            df_mean[["row_uid", "y_pred_blend"]].to_parquet(outdir / "test_blend.parquet", index=False)
            
        if "y_pred_stack" in df_mean.columns:
            df_mean[["row_uid", "y_pred_stack"]].to_parquet(outdir / "test_stack.parquet", index=False)

        metrics_conso = {}
        if "logS" in df_mean.columns:
            y_true = df_mean["logS"].values
            
            # Calcular métricas sobre el promedio (Consenso)
            for m in MODELS:
                if m in df_mean.columns:
                    metrics_conso[m] = get_metrics(y_true, df_mean[m].values)
            
            # Calcular Physio Range sobre el promedio
            if "temp_C" in df_mean.columns:
                t_val = df_mean["temp_C"].values
                mask_phys = (t_val >= 35) & (t_val <= 38)
                if mask_phys.any():
                    metrics_conso["physio_range"] = {}
                    for m in MODELS:
                        if m in df_mean.columns:
                            metrics_conso["physio_range"][m] = get_metrics(y_true[mask_phys], df_mean.loc[mask_phys, m].values)

        # Guardar archivo JSON obligatorio para Nextflow
        json_path = outdir / "metrics_cv_consensus.json"
        with open(json_path, "w") as f:
            json.dump(metrics_conso, f, indent=2)
        
        print(f"[CV] JSON guardado en: {json_path}")

if __name__ == "__main__":
    main()