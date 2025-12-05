#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_oof_chemprop.py (Pure Graph Strategy)
-------------------------------------------
Entrena Chemprop usando SOLO la estructura (SMILES).
Ignora explícitamente cualquier descriptor extra para evitar ruido.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.pruners import HyperbandPruner, NopPruner

def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet": return pd.read_parquet(p)
    elif p.suffix.lower() == ".csv": return pd.read_csv(p)
    raise ValueError(f"Formato no soportado: {p.suffix}")

def _metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "r2": float(r2), "n": int(len(y_true))}

def _run(cmd, check=True):
    print("CMD:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=check)

def _chemprop_train(train_csv, val_csv, out_dir, hp, use_gpu, epochs, seed, batch_size, target_col, weight_col=None, use_weights=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    comb_csv = out_dir / "data.csv"
    weights_csv = out_dir / "weights.csv"
    val_dummy_csv = out_dir / "val_dummy.csv"
    
    df_tr = pd.read_csv(train_csv)
    
    # --- ESTRATEGIA GRAFO PURO ---
    # Solo guardamos SMILES y TARGET. Ignoramos el resto.
    cols_to_save = ["smiles_neutral", target_col]
    
    apply_weights = (weight_col is not None and use_weights)

    cmd = [
        "chemprop_train",
        "--dataset_type", "regression",
        "--save_dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--smiles_columns", "smiles_neutral",
        "--target_columns", target_col,
        "--metric", "rmse",
    ]

    if apply_weights:
        # Estrategia con Pesos (Dummy Val)
        if weight_col not in df_tr.columns: df_tr[weight_col] = 1.0
        
        df_tr[cols_to_save].to_csv(comb_csv, index=False)
        df_tr[[weight_col]].to_csv(weights_csv, index=False, header=True)
        df_tr.iloc[:5][cols_to_save].to_csv(val_dummy_csv, index=False)

        cmd += [
            "--data_path", str(comb_csv),
            "--separate_val_path", str(val_dummy_csv),
            "--separate_test_path", str(val_dummy_csv),
            "--data_weights_path", str(weights_csv)
        ]
    else:
        # Estrategia Optuna (Split Random)
        if val_csv:
             df_va = pd.read_csv(val_csv)
             df_comb = pd.concat([df_tr, df_va], ignore_index=True)
             df_comb[cols_to_save].to_csv(comb_csv, index=False)
             
             n_tot = len(df_comb); n_tr = len(df_tr)
             ftr = n_tr/n_tot; fva = 1.0 - ftr
             cmd += ["--data_path", str(comb_csv), "--split_type", "random", "--split_sizes", f"{ftr:.4f}", f"{fva:.4f}", "0.0"]
        else:
             df_tr[cols_to_save].to_csv(comb_csv, index=False)
             cmd += ["--data_path", str(comb_csv), "--split_type", "random", "--split_sizes", "0.8", "0.2", "0.0"]

    if use_gpu: cmd += ["--gpu", "0"]

    if "message_hidden_dim" in hp: cmd += ["--hidden_size", str(int(hp["message_hidden_dim"]))]
    if "depth" in hp:              cmd += ["--depth", str(int(hp["depth"]))]
    if "dropout" in hp:            cmd += ["--dropout", str(float(hp["dropout"]))]
    if "ffn_num_layers" in hp:     cmd += ["--ffn_num_layers", str(int(hp["ffn_num_layers"]))]
    
    # NO añadimos --no_features_scaling porque NO hay features extra.

    _run(cmd, check=True)

def _chemprop_predict(val_in_csv, model_path, out_csv, gpu, batch_size, smiles_col):
    df = pd.read_csv(val_in_csv)
    tmp_in = Path(out_csv).parent / "pred_in.csv"
    
    # Solo SMILES para predecir
    df[[smiles_col]].to_csv(tmp_in, index=False)

    cmd = [
        "chemprop_predict",
        "--test_path", str(tmp_in),
        "--preds_path", str(out_csv),
        "--checkpoint_path", str(model_path),
        "--batch_size", str(batch_size),
        "--smiles_columns", smiles_col
    ]
    
    if gpu: cmd += ["--gpu", "0"]
    _run(cmd, check=True)

def _select_best_pt(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    ckpts = sorted(list(run_dir.rglob("*.pt")))
    if not ckpts: raise FileNotFoundError(f"No .pt found in {run_dir}")
    best = [c for c in ckpts if c.name == "model.pt"]
    return best[0] if best else ckpts[0]

def _hp_sample_optuna(trial):
    return {
        "message_hidden_dim": trial.suggest_int("message_hidden_dim", 300, 800, step=100),
        "depth": trial.suggest_int("depth", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 2)
    }
    
def _default_rungs(max_epochs): return [int(max_epochs*0.25), int(max_epochs*0.5), max_epochs]
def _get_pruner(name): return NopPruner()

def _write_fold_csv(df_tr, df_va, smiles_col, target, out_train, out_val):
    cols = [smiles_col, target]
    if "sw_temp37" in df_tr.columns: cols.append("sw_temp37")
    df_tr[cols].to_csv(out_train, index=False)
    df_va[cols].to_csv(out_val, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--smiles-col", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--weight-col", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", default=False)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tune-trials", type=int, default=0)
    ap.add_argument("--tune-timeout", type=int, default=None)
    ap.add_argument("--tune-pruner", default="asha")
    ap.add_argument("--asha-rungs", nargs="*", type=int, default=None)
    # Compat args
    ap.add_argument("--checkpoint", default=None)

    args = ap.parse_args()
    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)
    
    df = _read(args.train).copy()
    folds = _read(args.folds).copy()
    df = df.merge(folds[[args.id_col,"fold"]], on=args.id_col, how="inner")
    if args.weight_col and args.weight_col not in df.columns: df[args.weight_col] = 1.0

    oof = pd.DataFrame({args.id_col: df[args.id_col], "fold": df["fold"], "y_true": df[args.target]})
    oof["y_hat"] = np.nan
    
    folds_u = sorted(df["fold"].unique().tolist())
    rungs = sorted(args.asha_rungs) if args.asha_rungs else _default_rungs(args.epochs)
    best_by_fold = {}

    for k in folds_u:
        print(f"\n========== FOLD {k} ==========")
        fold_dir = outdir / f"fold_{k}"
        if fold_dir.exists(): shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)

        tr = df[df["fold"] != k].copy()
        va = df[df["fold"] == k].copy()
        
        tr_csv = fold_dir / "train.csv"
        va_csv = fold_dir / "val.csv"
        _write_fold_csv(tr, va, args.smiles_col, args.target, tr_csv, va_csv)

        if int(args.tune_trials) > 0:
            study = optuna.create_study(direction="minimize", pruner=_get_pruner(args.tune_pruner))
            def objective(trial):
                hp = _hp_sample_optuna(trial)
                from sklearn.model_selection import train_test_split
                tri, vai = train_test_split(tr, test_size=0.2, random_state=args.seed)
                
                tdir = fold_dir / f"trial_{trial.number}"
                if tdir.exists(): shutil.rmtree(tdir)
                tdir.mkdir(parents=True, exist_ok=True)
                
                tri_csv = tdir / "tri.csv"; vai_csv = tdir / "vai.csv"
                _write_fold_csv(tri, vai, args.smiles_col, args.target, tri_csv, vai_csv)
                
                best_rmse = None
                try:
                    for r_ep in rungs:
                        r_dir = tdir / f"e{r_ep}"
                        _chemprop_train(tri_csv, vai_csv, r_dir, hp, args.gpu, r_ep, args.seed, args.batch_size, args.target, args.weight_col, use_weights=False)
                        
                        pt = _select_best_pt(r_dir)
                        p_out = r_dir / "pred.csv"
                        _chemprop_predict(vai_csv, pt, p_out, args.gpu, args.batch_size, args.smiles_col)
                        
                        pred = pd.read_csv(p_out)
                        if args.target in pred.columns: vals = pred[args.target].values
                        else: vals = pred.iloc[:,0].values
                        rmse = mean_squared_error(vai[args.target].values, vals, squared=False)
                        best_rmse = rmse
                        trial.report(rmse, r_ep)
                        if trial.should_prune(): raise optuna.TrialPruned()
                finally:
                    shutil.rmtree(tdir, ignore_errors=True)
                return best_rmse
            
            study.optimize(objective, n_trials=int(args.tune_trials))
            
            # Recuperación directa y segura de parámetros
            print(f"[Optuna] Best params: {study.best_trial.params}")
            params = study.best_trial.params
            
            best_hp = {
                "message_hidden_dim": params.get("message_hidden_dim", 300),
                "depth": params.get("depth", 3),
                "dropout": params.get("dropout", 0.0),
                "ffn_num_layers": params.get("ffn_num_layers", 2)
            }
            
            best_by_fold[k] = best_hp
        else:
            best_hp = {"message_hidden_dim": 300, "depth": 3, "dropout": 0.0, "ffn_num_layers": 2}
            best_by_fold[k] = best_hp
            
        fin_dir = fold_dir / "final"
        _chemprop_train(tr_csv, None, fin_dir, best_hp, args.gpu, args.epochs, args.seed, args.batch_size, args.target, args.weight_col, use_weights=True)
        
        pt = _select_best_pt(fin_dir)
        oof_out = fold_dir / "oof_pred.csv"
        _chemprop_predict(va_csv, pt, oof_out, args.gpu, args.batch_size, args.smiles_col)
        
        pred = pd.read_csv(oof_out)
        if args.target in pred.columns: vals = pred[args.target].values
        else: vals = pred.iloc[:,0].values
        oof.loc[oof["fold"]==k, "y_hat"] = vals

    common = pd.DataFrame({"id": df[args.id_col], "fold": df["fold"], "y_true": df[args.target], "y_pred": oof["y_hat"]})
    common.to_parquet(outdir / "chemprop.parquet", index=False)
    
    with open(outdir / "chemprop_best_params.json", "w") as f:
        json.dump(list(best_by_fold.values())[0], f, indent=2)
    
    m = _metrics(common["y_true"], common["y_pred"])
    with open(outdir / "metrics_oof_chemprop.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"[DONE] OOF RMSE: {m['rmse']:.4f}")

if __name__ == "__main__":
    main()