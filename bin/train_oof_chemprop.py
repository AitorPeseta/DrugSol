#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner, NopPruner

def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Formato no soportado: {p.suffix} ({path})")

def _metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "r2": float(r2), "n": int(len(y_true))}

def _run(cmd, check=True):
    print("CMD:", " ".join(map(str, cmd)))
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("[ERROR] returncode:", e.returncode)
        if e.stdout:
            print("--- STDOUT ---\n", e.stdout)
        if e.stderr:
            print("--- STDERR ---\n", e.stderr)
        # re-lanza para que Nextflow detecte fallo
        raise

def _chemprop_train(train_csv, val_csv, out_dir, hp, use_gpu, epochs, seed, batch_size):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    # Archivos de salida
    comb_csv = out_dir / "combined.csv"
    weights_csv = out_dir / "weights.csv"  # <--- NUEVO ARCHIVO PARA PESOS
    splits_json = out_dir / "splits.json"

    # Cargar y combinar
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    
    # 1. ASEGURARSE DE QUE EXISTE LA COLUMNA DE PESOS
    # Si por alguna razón no existe, rellenamos con 1.0
    if "sw_temp37" not in df_tr.columns:
        df_tr["sw_temp37"] = 1.0
    if "sw_temp37" not in df_va.columns:
        df_va["sw_temp37"] = 1.0 # En validación el peso no afecta al entrenamiento, pero Chemprop pide que cuadren las filas
        
    df_comb = pd.concat([df_tr, df_va], ignore_index=True)
    
    # 2. GUARDAR LOS DATOS (SMILES + TARGET)
    df_comb.to_csv(comb_csv, index=False)

    # 3. GUARDAR LOS PESOS EN UN CSV APARTE
    # Chemprop espera un CSV donde cada fila i corresponde a la fila i del input
    # Puede tener cabecera o no, pero mejor ponerle una simple.
    df_comb[["sw_temp37"]].to_csv(weights_csv, index=False)

    # Índices de train/val
    n_tr = len(df_tr); n_va = len(df_va)
    splits = [{
        "train": list(range(0, n_tr)),
        "val": list(range(n_tr, n_tr + n_va)),
        "test": []
    }]
    splits_json.write_text(json.dumps(splits))

    # Detectar columnas extra
    present_cols = [c for c in ["temp_C","n_ionizable","n_acid","n_base",
                                "TPSA","logP","HBD","HBA","FractionCSP3","MW", 
                                "inv_temp"] # <--- No olvides inv_temp si la tienes
                    if c in df_comb.columns]

    # Construir comando
    chemprop_bin = Path(sys.executable).parent / "chemprop"
    
    base = [
        str(chemprop_bin), "train",
        "-i", str(comb_csv),
        "--splits-file", str(splits_json),
        "--data-weights-path", str(weights_csv),  # <--- AQUÍ ESTÁ LA MAGIA
        "--smiles-columns","smiles_neutral",
        "--target-columns","logS",
        "-o", str(out_dir),
        "--epochs", str(epochs),
        "--metric","rmse",
        "--data-seed", str(seed),
        "--batch-size", str(batch_size),
    ]

    if present_cols:
        base += ["--descriptors-columns", *present_cols]

    if use_gpu:
        base += ["--accelerator","gpu","--devices","1"]

    # HP de Optuna
    if "message_hidden_dim" in hp: base += ["--message-hidden-dim", str(int(hp["message_hidden_dim"]))]
    if "depth" in hp:              base += ["--depth", str(int(hp["depth"]))]
    if "dropout" in hp:            base += ["--dropout", str(float(hp["dropout"]))]
    if "ffn_num_layers" in hp:     base += ["--ffn-num-layers", str(int(hp["ffn_num_layers"]))]

    r = _run(base, check=True)
    if r.stdout: print(r.stdout)
    if r.stderr: print(r.stderr)


def _chemprop_predict(val_in_csv: Path, model_path: Path, out_csv: Path, gpu: bool, batch_size: int):
    df_val = pd.read_csv(val_in_csv)
    out_dir = Path(out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Solo columnas presentes en el MISMO CSV de entrada
    present_cols = [c for c in ["temp_C","n_ionizable","n_acid","n_base",
                                "TPSA","logP","HBD","HBA","FractionCSP3","MW"]
                    if c in df_val.columns]

    chemprop_bin = Path(sys.executable).parent / "chemprop"
    cmd = [str(chemprop_bin), "predict",
           "-i", str(val_in_csv),
           "-o", str(out_csv),
           "--model-paths", str(model_path),
           "--drop-extra-columns",
           "--batch-size", str(batch_size)]

    if present_cols:
        cmd += ["--descriptors-columns", *present_cols]

    if gpu:
        cmd += ["--accelerator","gpu","--devices","1"]

    r = _run(cmd, check=True)
    if r.stdout: print(r.stdout)
    if r.stderr: print(r.stderr)

def _select_best_pt(run_dir: Path) -> Path:
    """Escoge el mejor checkpoint (mínimo val_loss o similar) entre ficheros .pt en save_dir."""
    run_dir = Path(run_dir)
    ckpts = sorted([p for p in run_dir.glob("*.pt")] + [p for p in run_dir.glob("**/*.pt")])
    if not ckpts:
        best = run_dir / "best.pt"
        if best.exists():
            return best
        cands = list(run_dir.rglob("*.pt"))
        if not cands:
            raise FileNotFoundError(f"No se encontraron checkpoints .pt en {run_dir}")
        return sorted(cands)[0]
    bests = [p for p in ckpts if p.name == "best.pt"]
    return bests[0] if bests else ckpts[0]

def _hp_sample_optuna(trial: optuna.Trial):
    hp = dict(
        message_hidden_dim = trial.suggest_int("message_hidden_dim", 800, 2400, step=40),
        depth              = trial.suggest_int("depth", 2, 6),
        dropout            = trial.suggest_float("dropout", 0.0, 0.2),
        ffn_num_layers     = trial.suggest_int("ffn_num_layers", 1, 3)
    )
    trial.set_user_attr("hp", hp)
    return hp

def _default_rungs(max_epochs: int):
    # tres rungs por defecto: 25%, 50% y 100%
    e1 = max(1, int(round(max_epochs * 0.25)))
    e2 = max(2, int(round(max_epochs * 0.50)))
    e3 = max_epochs
    return sorted(set([e1, e2, e3]))

def _get_pruner(name: str):
    name = (name or "asha").lower()
    if name in ("asha","hyperband"): return HyperbandPruner()
    if name in ("sha","successive_halving"): return SuccessiveHalvingPruner()
    if name in ("none","nop","off"): return NopPruner()
    return HyperbandPruner()

def _write_fold_csv(df_tr, df_va, smiles_col, target, out_train: Path, out_val: Path):
    # sanity + no-NaN
    needed = [smiles_col, target, "temp_C"]
    for c in needed:
        if c not in df_tr.columns or c not in df_va.columns:
            raise ValueError(f"Falta columna requerida en train/val: {c}")
    tr = df_tr.copy()
    va = df_va.copy()
    tr = tr[[smiles_col, target, "temp_C"]].dropna()
    va = va[[smiles_col, target, "temp_C"]].dropna()
    if tr.empty or va.empty:
        raise RuntimeError(f"Fold vacío tras dropna(): train={len(tr)} val={len(va)}")
    tr.to_csv(out_train, index=False)
    va.to_csv(out_val, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train parquet/csv con columnas: id, smiles, target, temp_C")
    ap.add_argument("--folds", required=True, help="csv/parquet con columnas: id_col, fold")
    ap.add_argument("--smiles-col", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", default=False)
    ap.add_argument("--batch-size", type=int, default=32)

    # Optuna / Tuning
    ap.add_argument("--tune-trials", type=int, default=0)
    ap.add_argument("--tune-timeout", type=int, default=None)
    ap.add_argument("--tune-pruner", default="asha")
    ap.add_argument("--asha-rungs", nargs="*", type=int, default=None)

    args = ap.parse_args()

    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = _read(args.train).copy()
    folds = _read(args.folds).copy()

    for c in (args.id_col, args.smiles_col, args.target, "temp_C"):
        if c not in df.columns: raise ValueError(f"Falta columna en train: {c}")
    for c in (args.id_col, "fold"):
        if c not in folds.columns: raise ValueError(f"Falta columna en folds: {c}")

    df = df.merge(folds[[args.id_col,"fold"]], on=args.id_col, how="inner")
    if df["fold"].isna().any(): raise ValueError("Hay filas sin fold tras el merge.")

    oof = pd.DataFrame({args.id_col: df[args.id_col], "fold": df["fold"]})
    oof["y_hat"] = np.nan

    folds_u = sorted(df["fold"].unique().tolist())
    print(f"[INFO] Folds únicos: {folds_u}")

    rungs = sorted(args.asha_rungs) if args.asha_rungs else _default_rungs(args.epochs)
    print(f"[ASHA] Rungs (épocas): {rungs}")

    # Para best_params.json global
    best_by_fold = {}
    rmse_by_fold = {}

    for k in folds_u:
        print(f"\n========== FOLD {k} ==========")
        fold_dir = outdir / f"fold_{k}"
        if fold_dir.exists(): shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)

        tr = df[df["fold"] != k].copy()
        va = df[df["fold"] == k].copy()

        train_csv = fold_dir / "train.csv"
        val_csv   = fold_dir / "val.csv"
        _write_fold_csv(tr, va, args.smiles_col, args.target, train_csv, val_csv)

        # Tuning por ASHA/Hyperband si se pide
        if int(args.tune_trials) > 0:
            pruner = _get_pruner(args.tune_pruner)
            study = optuna.create_study(direction="minimize",
                                        study_name=f"chemprop_fold_{k}",
                                        pruner=pruner)
            def objective(trial: optuna.Trial):
                hp = _hp_sample_optuna(trial)
                best_rmse = None
                 # --- split interno: de 'tr' sacamos tr_in/va_in para evaluar trials ---
                from sklearn.model_selection import train_test_split
                tr_in, va_in = train_test_split(tr, test_size=0.2, random_state=args.seed, shuffle=True)
                # archivos para el run interno
                tmp_dir_root = fold_dir / f"trial_{trial.number}"
                if tmp_dir_root.exists(): shutil.rmtree(tmp_dir_root)
                tmp_dir_root.mkdir(parents=True, exist_ok=True)
                tr_in_csv = tmp_dir_root / "train_in.csv"
                va_in_csv = tmp_dir_root / "val_in.csv"
                _write_fold_csv(tr_in, va_in, args.smiles_col, args.target, tr_in_csv, va_in_csv)
                for rung_epochs in rungs:
                    tmp_dir = tmp_dir_root / f"e{rung_epochs}"
                    if tmp_dir.exists(): shutil.rmtree(tmp_dir)
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        _chemprop_train(tr_in_csv, va_in_csv, tmp_dir, hp, args.gpu, rung_epochs, args.seed, args.batch_size)
                        best_pt = _select_best_pt(tmp_dir)
                        # PRED en la valid interna (no usar va del fold aquí)
                        pred_in = tmp_dir / "val_in.csv"   # reusamos va_in_csv
                        pred_out = tmp_dir / "val_pred.csv"
                        va_pred = pd.DataFrame({
                            "smiles": va_in[args.smiles_col].values,
                            "temp_C": va_in["temp_C"].values
                        })
                        va_pred.to_csv(pred_in, index=False)

                        _chemprop_predict(pred_in, best_pt, pred_out, args.gpu, args.batch_size)
                        pred = pd.read_csv(pred_out)

                        # Elegir columna correcta de predicción
                        cols_lower = {c.lower(): c for c in pred.columns}
                        if "prediction" in cols_lower:
                            pred_col = cols_lower["prediction"]
                        elif args.target.lower() in cols_lower:
                            pred_col = cols_lower[args.target.lower()]
                        elif "value" in cols_lower:
                            pred_col = cols_lower["value"]
                        else:
                            candidates = [c for c in pred.columns if c.lower() not in ("smiles", "temp_c", "temp_C")]
                            pred_col = candidates[-1] if candidates else pred.columns[-1]

                        yhat = pred[pred_col].values
                        rmse = mean_squared_error(va_in[args.target].values, yhat, squared=False)
                        best_rmse = rmse if best_rmse is None else min(best_rmse, rmse)
                        trial.report(rmse, step=rung_epochs)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    finally:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                shutil.rmtree(tmp_dir_root, ignore_errors=True)
                return best_rmse if best_rmse is not None else 9e9

            study.optimize(objective, n_trials=int(args.tune_trials), timeout=args.tune_timeout)
            best_hp = study.best_trial.user_attrs["hp"]
            print(f"[TUNE] Fold {k} best RMSE={study.best_value:.4f} with HP={best_hp}")
            best_by_fold[k] = dict(best_hp)
        else:
            best_hp = dict(message_hidden_dim=1600, depth=3, dropout=0.10, ffn_num_layers=1)
            print(f"[HP] Fold {k} (fixed): {best_hp}")
            best_by_fold[k] = dict(best_hp)

        # Entrenamiento final del fold
        final_dir = fold_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        _chemprop_train(train_csv, val_csv, final_dir, best_hp, args.gpu, args.epochs, args.seed, args.batch_size)
        best_pt = _select_best_pt(final_dir)

        # Pred de validación del fold
        pred_in_csv  = fold_dir / "val_in.csv"
        pred_out_csv = fold_dir / "val_pred.csv"
        va_pred = pd.DataFrame({
            "smiles": va[args.smiles_col].values,
            "temp_C": va["temp_C"].values
        })
        va_pred.to_csv(pred_in_csv, index=False)
        _chemprop_predict(pred_in_csv, best_pt, pred_out_csv, args.gpu, args.batch_size)

        pred = pd.read_csv(pred_out_csv)
        cols_lower = {c.lower(): c for c in pred.columns}
        if "prediction" in cols_lower:
            pred_col = cols_lower["prediction"]
        elif args.target.lower() in cols_lower:
            pred_col = cols_lower[args.target.lower()]
        elif "value" in cols_lower:
            pred_col = cols_lower["value"]
        else:
            candidates = [c for c in pred.columns if c.lower() not in ("smiles", "temp_C")]
            pred_col = candidates[-1] if candidates else pred.columns[-1]

        yhat = pred[pred_col].values

        m_fold = _metrics(va[args.target].values, yhat)
        (fold_dir / "metrics_fold.json").write_text(json.dumps(m_fold, indent=2))
        (fold_dir / "best_hp.json").write_text(json.dumps(best_hp, indent=2))
        print(f"[OK] Fold {k} metrics:", m_fold)
        rmse_by_fold[k] = m_fold["rmse"]

        # Guardar OOF para este fold
        oof.loc[oof["fold"] == k, "y_hat"] = yhat

    # === Métricas OOF ===
    merged = df[[args.id_col, args.target]].merge(
        oof[[args.id_col, "y_hat"]], on=args.id_col, how="left"
    )
    # añadir fold para el "formato común"
    merged = merged.merge(df[[args.id_col, "fold"]], on=args.id_col, how="left")
    m_all = _metrics(merged[args.target].values, merged["y_hat"].values)
    print("[OK] OOF metrics:", m_all)

    # === Formato común para meta_stack_blend ===
    common = merged[[args.id_col, "fold", args.target, "y_hat"]].copy()
    common.columns = ["id", "fold", "y_true", "y_pred"]
    # asegurar tipos
    common["fold"] = common["fold"].astype(int)
    oof_common_path = outdir / "chemprop.parquet"  # <- el stacker lee este
    common.to_parquet(oof_common_path, index=False)
    print(f"[OK] Guardado OOF (formato común): {oof_common_path.resolve()}")

    # === Formato legacy/interno (por si lo usa otra etapa) ===
    oof_legacy = oof.rename(columns={"y_hat": "y_chemprop_oof"}).drop(columns=["fold"])
    oof_legacy_path = outdir / "chemprop_oof.parquet"
    oof_legacy.to_parquet(oof_legacy_path, index=False)
    print(f"[OK] Guardado OOF (legacy): {oof_legacy_path.resolve()}")

    # === Métricas: ambos nombres por compatibilidad ===
    metrics_new    = outdir / "metrics_oof_chemprop.json"
    metrics_legacy = outdir / "metrics_oof.json"
    payload = json.dumps(m_all, indent=2)
    metrics_new.write_text(payload)
    metrics_legacy.write_text(payload)
    print(f"[OK] Guardado métricas OOF: {metrics_new.resolve()}")
    print(f"[OK] Guardado métricas OOF (legacy): {metrics_legacy.resolve()}")

    # ---- Resumen global de mejores HParams (formato legacy) ----
    if best_by_fold:
        # Elegimos el fold con menor RMSE
        best_fold = min(rmse_by_fold, key=rmse_by_fold.get)
        hp_best = best_by_fold[best_fold]
        best_params_global = {
            "batch_size": args.batch_size,
            "hidden_size": int(hp_best["message_hidden_dim"]),
            "depth": int(hp_best["depth"]),
            "dropout": float(hp_best["dropout"]),
            "ffn_num_layers": int(hp_best["ffn_num_layers"]),
            "epochs": args.epochs,
        }
        (outdir / "chemprop_best_params.json").write_text(json.dumps(best_params_global, indent=2))
        print(f"[OK] Guardado resumen global de HP: {(outdir / 'chemprop_best_params.json').resolve()}")

if __name__ == "__main__":
    main()
