#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_oof_chemprop.py (Robust v1.6.1 Fix)
"""

import argparse, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner, NopPruner
from rdkit import Chem # Para validar SMILES

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

def is_valid_smiles(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return m is not None
    except:
        return False

def _chemprop_train(train_csv, val_csv, out_dir, hp, use_gpu, epochs, seed, batch_size, target_col, weight_col=None, use_weights=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar y Combinar
    df_tr = pd.read_csv(train_csv)
    
    # Si no estamos en modo weights (Optuna), combinamos con val para tener más datos
    # Si estamos en modo weights (Final), ya le pasamos todo lo que queremos usar en train_csv
    if not use_weights and val_csv:
         df_va = pd.read_csv(val_csv)
         df = pd.concat([df_tr, df_va], ignore_index=True)
    else:
         df = df_tr.copy()

    # 2. SANEAMIENTO CRÍTICO: Filtrar SMILES inválidos AHORA
    # Esto evita que Chemprop los borre silenciosamente y desalinee los pesos
    valid_mask = df["smiles_neutral"].apply(is_valid_smiles)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"[WARN] Descartados {n_dropped} SMILES inválidos antes de entrenar para asegurar alineación.")
        df = df[valid_mask].reset_index(drop=True)

    # 3. Preparar Archivos
    data_path = out_dir / "data.csv"
    weights_path = out_dir / "weights.csv"
    
    # Columnas
    cols_data = ["smiles_neutral", target_col]
    exclude = ["smiles_neutral", target_col, "row_uid", "fold", "smiles", "mol", weight_col or ""]
    desc_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Guardar Datos
    df[cols_data + desc_cols].to_csv(data_path, index=False)

    cmd = [
        "chemprop_train",
        "--data_path", str(data_path),
        "--dataset_type", "regression",
        "--save_dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--smiles_columns", "smiles_neutral",
        "--target_columns", target_col,
        "--metric", "rmse",
    ]

    # 4. Gestión de Pesos (Con Header y Split Automático)
    if weight_col and use_weights:
        # Asegurar que existe
        if weight_col not in df.columns: df[weight_col] = 1.0
        # Guardar con HEADER=True (estándar)
        df[[weight_col]].to_csv(weights_path, index=False, header=True)
        cmd += ["--data_weights_path", str(weights_path)]
        
        # TRUCO: Usar un split minúsculo para validación (1%)
        # Esto evita bugs de "no validation data" y permite que Chemprop gestione los índices de pesos
        cmd += ["--split_type", "random", "--split_sizes", "0.99", "0.01", "0.0"]
    
    else:
        # Sin pesos (Optuna), split 80/20 estándar
        cmd += ["--split_type", "random", "--split_sizes", "0.8", "0.2", "0.0"]

    if use_gpu:
        cmd += ["--gpu", "0"]

    if "message_hidden_dim" in hp: cmd += ["--hidden_size", str(int(hp["message_hidden_dim"]))]
    if "depth" in hp:              cmd += ["--depth", str(int(hp["depth"]))]
    if "dropout" in hp:            cmd += ["--dropout", str(float(hp["dropout"]))]
    if "ffn_num_layers" in hp:     cmd += ["--ffn_num_layers", str(int(hp["ffn_num_layers"]))]
    
    if desc_cols:
        cmd += ["--no_features_scaling"]

    _run(cmd, check=True)

def _chemprop_predict(val_in_csv, model_path, out_csv, gpu, batch_size, smiles_col):
    df = pd.read_csv(val_in_csv)
    tmp_in = Path(out_csv).parent / "pred_in.csv"
    
    exclude = [smiles_col, "logS", "row_uid", "fold", "smiles", "mol"]
    desc_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    df[[smiles_col] + desc_cols].to_csv(tmp_in, index=False)

    cmd = [
        "chemprop_predict",
        "--test_path", str(tmp_in),
        "--preds_path", str(out_csv),
        "--checkpoint_path", str(model_path),
        "--batch_size", str(batch_size),
        "--smiles_columns", smiles_col
    ]
    
    if desc_cols:
        cmd += ["--no_features_scaling"]

    if gpu:
        cmd += ["--gpu", "0"]

    _run(cmd, check=True)

def _select_best_pt(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    ckpts = sorted(list(run_dir.rglob("*.pt")))
    if not ckpts:
         if (run_dir / "model.pt").exists(): return run_dir / "model.pt"
         raise FileNotFoundError(f"No .pt found in {run_dir}")
    best = [c for c in ckpts if c.name == "model.pt"]
    return best[0] if best else ckpts[0]

def _hp_sample_optuna(trial: optuna.Trial):
    hp = dict(
        message_hidden_dim = trial.suggest_int("message_hidden_dim", 300, 600, step=100),
        depth              = trial.suggest_int("depth", 2, 4),
        dropout            = trial.suggest_float("dropout", 0.0, 0.2),
        ffn_num_layers     = trial.suggest_int("ffn_num_layers", 1, 2)
    )
    trial.set_user_attr("hp", hp)
    return hp

def _default_rungs(max_epochs: int):
     e1 = max(1, int(round(max_epochs * 0.25)))
     e2 = max(2, int(round(max_epochs * 0.50)))
     e3 = max_epochs
     return sorted(set([e1, e2, e3]))

def _get_pruner(name: str):
     return NopPruner() # Simplificamos para evitar líos

def _write_fold_csv(df_tr, df_va, smiles_col, target, out_train: Path, out_val: Path):
     needed = [smiles_col, target]
     for c in needed:
         if c not in df_tr.columns or c not in df_va.columns:
             raise ValueError(f"Falta columna requerida en train/val: {c}")
     df_tr.to_csv(out_train, index=False)
     df_va.to_csv(out_val, index=False)

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

    args = ap.parse_args()
    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)
    
    df = _read(args.train).copy()
    folds = _read(args.folds).copy()
    df = df.merge(folds[[args.id_col,"fold"]], on=args.id_col, how="inner")
    
    if args.weight_col and args.weight_col not in df.columns:
        df[args.weight_col] = 1.0

    oof = pd.DataFrame({args.id_col: df[args.id_col], "fold": df["fold"]})
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
                        # Optuna: Sin pesos, solo datos
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
            best_hp = study.best_trial.user_attrs["hp"]
            best_by_fold[k] = best_hp
        else:
            best_hp = {"message_hidden_dim": 300, "depth": 3, "dropout": 0.0, "ffn_num_layers": 2}
            best_by_fold[k] = best_hp
            
        # Final Train (Con pesos)
        fin_dir = fold_dir / "final"
        # Pasamos 'tr' y 'None' (no validación explícita, se usa split interno 99/1)
        # Esto es CLAVE: No pasamos 'va' a train, solo 'tr' para que aprenda de todo.
        # Validamos luego con 'va' en el predict.
        tr_csv_full = fold_dir / "train_full_fold.csv"
        tr.to_csv(tr_csv_full, index=False)
        
        _chemprop_train(tr_csv_full, None, fin_dir, best_hp, args.gpu, args.epochs, args.seed, args.batch_size, args.target, args.weight_col, use_weights=True)
        
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