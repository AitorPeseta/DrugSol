#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOF para XGB y LGBM con folds precomputados + Optuna (nested CV sin fuga) + ASHA.
XGBoost (2.1.* compatible): entrena con API nativa xgb.train + early stopping.

Salidas:
  - oof_gbm/oof/xgb.parquet  (id, fold, y_true, y_pred, model, seed)
  - oof_gbm/oof/lgbm.parquet
  - oof_gbm/metrics_tree.json
  - oof_gbm/hp/xgb_fold{k}.json, oof_gbm/hp/lgbm_fold{k}.json (mejores HP por fold)
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

import xgboost as xgb
import lightgbm as lgb
from lightgbm.basic import LightGBMError

# Optuna (opcional)
try:
    import optuna
    from optuna.exceptions import TrialPruned
except Exception:
    optuna = None
    TrialPruned = Exception

RANDOM_STATE = 42

# -------------------- Args --------------------

def parse_args():
    ap = argparse.ArgumentParser("OOF para XGB y LGBM con folds precomputados + Optuna + ASHA")
    ap.add_argument("--train", required=True, help="Train tabular (parquet/csv) con features numéricas y target")
    ap.add_argument("--folds", required=True, help="folds.parquet con columnas [id, fold]")
    ap.add_argument("--id-col", dest="id_col", default="row_uid",
                    help="Nombre de la columna id en train/folds (default: row_uid)")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", dest="save_dir", default="./oof_gbm")
    ap.add_argument("--use-gpu", dest="use_gpu", action="store_true",
                    help="XGB/LGBM en GPU si disponible (XGB usa gpu_hist)")

    # hiperparámetros base si no hay tuning
    ap.add_argument("--xgb-n-estimators", dest="xgb_n_estimators", type=int, default=2000)
    ap.add_argument("--xgb-learning-rate", dest="xgb_learning_rate", type=float, default=0.03)
    ap.add_argument("--lgbm-n-estimators", dest="lgbm_n_estimators", type=int, default=3000)
    ap.add_argument("--lgbm-learning-rate", dest="lgbm_learning_rate", type=float, default=0.03)

    # Optuna
    ap.add_argument("--tune-trials", dest="tune_trials", type=int, default=0,
                    help=">0 activa Optuna nested por fold")
    ap.add_argument("--inner-splits", dest="inner_splits", type=int, default=3,
                    help="nº de folds internos para Optuna")

    # PRUNER (ASHA)
    ap.add_argument("--pruner", choices=["asha", "none"], default="asha",
                    help="Pruner para Optuna (default: asha)")
    ap.add_argument("--asha-min-resource", type=int, default=1,
                    help="Recurso mínimo para ASHA (usamos 'folds' como pasos; default=1)")
    ap.add_argument("--asha-reduction-factor", type=int, default=3,
                    help="Factor de reducción de ASHA (default=3)")
    ap.add_argument("--asha-min-early-stopping-rate", type=int, default=0,
                    help="min_early_stopping_rate de ASHA (default=0)")

    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    return ap.parse_args()

# -------------------- Utils --------------------

def rmse(y, yhat): return float(np.sqrt(mean_squared_error(y, yhat)))
def r2(y, yhat):   return float(r2_score(y, yhat))

def load_and_merge(train_path, folds_path, id_col, target):
    tr = pd.read_parquet(train_path) if str(train_path).endswith(".parquet") else pd.read_csv(train_path)
    folds = pd.read_parquet(folds_path) if str(folds_path).endswith(".parquet") else pd.read_csv(folds_path)

    if id_col not in tr.columns:
        raise ValueError(f"Falta '{id_col}' en train (pásalo con --id-col si no es 'row_uid').")
    if target not in tr.columns:
        raise ValueError(f"Falta target '{target}' en train")
    if id_col not in folds.columns or "fold" not in folds.columns:
        raise ValueError(f"folds debe tener columnas [{id_col}, fold]")

    df = tr.merge(folds[[id_col, "fold"]], on=id_col, how="inner").reset_index(drop=True)
    if df.empty:
        raise ValueError("Tras el merge con folds no quedan filas.")
    df = df[~df[target].isna()].reset_index(drop=True)

    y = df[target].to_numpy()
    ids = df[id_col].to_numpy()
    folds_values = df["fold"].to_numpy()

    # solo numéricas
    num = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in [target, id_col, "fold"] if c in num.columns]
    if drop_cols:
        num = num.drop(columns=drop_cols)
    if num.shape[1] == 0:
        raise ValueError("No quedan columnas numéricas válidas tras filtrar.")

    return df, num, y, ids, folds_values, num.columns.tolist()

def make_preproc(num_cols):
    return ColumnTransformer(
        transformers=[(
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("const", VarianceThreshold(threshold=0.0)),
            ]),
            num_cols
        )],
        remainder="drop",
    )

# -------- XGBoost (API nativa con ES) --------

def _xgb_fit_predict_booster(X_tr, y_tr, X_va, y_va, params, use_gpu: bool, seed: int, esr: int = 100):
    """
    Entrena XGBoost con xgb.train + early stopping y devuelve (booster, preds_val).
    Compatible con XGB 2.x (sin ntree_limit).
    """
    booster_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": float(params.get("learning_rate", params.get("eta", 0.03))),
        "max_depth": int(params.get("max_depth", 8)),
        "min_child_weight": float(params.get("min_child_weight", 2.0)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "lambda": float(params.get("reg_lambda", params.get("lambda", 1.0))),
        "alpha": float(params.get("reg_alpha", params.get("alpha", 0.0))),
        "gamma": float(params.get("gamma", 0.0)),
        "verbosity": 0,
        "seed": int(seed),
    }
    if use_gpu:
        booster_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    else:
        booster_params.update({"tree_method": "hist"})

    num_round = int(params.get("n_estimators", 2000))

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)

    booster = xgb.train(
        booster_params,
        dtrain,
        num_boost_round=num_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=esr,
        verbose_eval=False,
    )

    best_it = getattr(booster, "best_iteration", None)
    if best_it is not None:
        y_hat = booster.predict(dvalid, iteration_range=(0, best_it + 1))
    else:
        y_hat = booster.predict(dvalid)

    return booster, y_hat


def _preprocess_train_val(X_tr_raw, X_va_raw, y_tr, num_cols):
    pre = make_preproc(num_cols)
    X_tr = pre.fit_transform(X_tr_raw)
    X_va = pre.transform(X_va_raw)
    return pre, X_tr, X_va

# -------------------- Model factories --------------------

def get_xgb_base_params(use_gpu: bool, seed: int, n_estimators: int, lr: float):
    return dict(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=8,
        min_child_weight=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        use_gpu=use_gpu,
        random_state=seed,
    )

def get_lgbm_model(use_gpu: bool, seed: int):
    params = dict(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=256,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        objective="regression",
        metric="rmse",
    )
    if use_gpu:
        params.update(dict(device_type="gpu"))
    return lgb.LGBMRegressor(**params)

# -------------------- CV helpers --------------------

def inner_cv_score(model_name, base_params, params, X_raw, y, inner_splits, num_cols, seed, use_gpu, trial=None):
    """
    RMSE medio en inner-CV (menor mejor).
    Si 'trial' no es None, reporta métricas por fold y permite pruning ASHA.
    """
    kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    rmses = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_raw), start=1):
        X_tr_raw, X_va_raw = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        pre, X_tr, X_va = _preprocess_train_val(X_tr_raw, X_va_raw, y_tr, num_cols)

        if model_name == "xgb":
            p = dict(base_params); p.update(params)
            _, y_hat = _xgb_fit_predict_booster(X_tr, y_tr, X_va, y_va, p, use_gpu, seed, esr=100)
        elif model_name == "lgbm":
            est = get_lgbm_model(use_gpu, seed)
            est.set_params(**params)
            est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                    callbacks=[lgb.early_stopping(100, verbose=False)])
            y_hat = est.predict(X_va, num_iteration=getattr(est, "best_iteration_", None))
        else:
            raise ValueError("model_name debe ser 'xgb' o 'lgbm'")

        rmses.append(rmse(y_va, y_hat))

        # --- Reporte incremental para ASHA ---
        if trial is not None:
            mean_so_far = float(np.mean(rmses))
            trial.report(mean_so_far, step=fold_idx)   # step = #folds evaluados
            if trial.should_prune():
                raise TrialPruned(f"Pruned at inner fold {fold_idx} (mean RMSE={mean_so_far:.4f})")

    return float(np.mean(rmses))

def fit_predict_one_fold(model_name, base_params, X_raw, y, folds_values, fold_k, num_cols, use_gpu, seed):
    """Entrena en ~k y predice en k (sin Optuna)."""
    va_mask = (folds_values == fold_k)
    tr_mask = ~va_mask
    X_tr_raw, X_va_raw = X_raw.loc[tr_mask, :], X_raw.loc[va_mask, :]
    y_tr, y_va = y[tr_mask], y[va_mask]
    pre, X_tr, X_va = _preprocess_train_val(X_tr_raw, X_va_raw, y_tr, num_cols)

    if model_name == "xgb":
        _, y_hat = _xgb_fit_predict_booster(X_tr, y_tr, X_va, y_va, base_params, use_gpu, seed, esr=100)
        return y_hat

    elif model_name == "lgbm":
        est = get_lgbm_model(use_gpu, seed)
        est.set_params(
            n_estimators=max(3000, est.get_params().get("n_estimators", 3000)),
            learning_rate=min(0.03, est.get_params().get("learning_rate", 0.03)),
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l1=0.0, lambda_l2=1.0,
        )
        est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        return est.predict(X_va, num_iteration=getattr(est, "best_iteration_", None))

    else:
        raise ValueError("model_name debe ser 'xgb' o 'lgbm'")

# -------------------- Optuna spaces --------------------

def suggest_space_xgb(trial):
    return dict(
        max_depth=trial.suggest_int("max_depth", 6, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        subsample=trial.suggest_float("subsample", 0.6, 0.9),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9),
        # L2 y L1: con escala log y límite inferior > 0
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        reg_alpha =trial.suggest_float("reg_alpha",  1e-8, 10.0, log=True),
        # tasa de aprendizaje opcional
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        n_estimators=5000,  # grande para permitir early_stopping/pruner
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
    )


def suggest_space_lgbm(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 64, 512),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 0.9),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 0.9),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 100),
        # regularización L1/L2 en escala log con low>0
        reg_alpha= trial.suggest_float("lgbm_reg_alpha",  1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("lgbm_reg_lambda", 1e-8, 10.0, log=True),
        learning_rate=trial.suggest_float("lgbm_lr", 1e-3, 3e-1, log=True),
        n_estimators=5000,  # grande
        objective="regression",
        metric="rmse",
    )


def tune_one_fold(model_name, base_params, X_raw, y, folds_values, fold_k, num_cols,
                  trials, inner_splits, seed, use_gpu, hp_outdir,
                  pruner_kind="asha", asha_min_resource=1, asha_reduction=3, asha_min_esr=0):
    """Tunea en train_k con inner-CV+ASHA, reentrena en train_k y predice val_k."""
    if optuna is None:
        raise SystemExit("[ERROR] Optuna no está instalado y --tune-trials>0.")

    va_mask = (folds_values == fold_k)
    tr_mask = ~va_mask
    X_tr_raw, y_tr = X_raw.loc[tr_mask, :], y[tr_mask]
    X_va_raw, y_va = X_raw.loc[va_mask, :], y[va_mask]

    # Pruner
    if pruner_kind == "asha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(asha_min_resource),
            reduction_factor=int(asha_reduction),
            min_early_stopping_rate=int(asha_min_esr),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    def objective(trial):
        params = suggest_space_xgb(trial) if model_name == "xgb" else suggest_space_lgbm(trial)
        # Inner-CV con reporte incremental (step = fold index)
        return inner_cv_score(
            model_name, base_params, params,
            X_tr_raw, y_tr, inner_splits, num_cols, seed, use_gpu, trial=trial
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params = study.best_params

    # Reentrena en train_k y predice en val_k
    pre, X_tr, X_va = _preprocess_train_val(X_tr_raw, X_va_raw, y_tr, num_cols)
    if model_name == "xgb":
        p = dict(base_params); p.update(best_params)
        _, y_hat = _xgb_fit_predict_booster(X_tr, y_tr, X_va, y_va, p, use_gpu, seed, esr=100)
    else:
        est = get_lgbm_model(use_gpu, seed)
        est.set_params(**best_params)
        est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        y_hat = est.predict(X_va, num_iteration=getattr(est, "best_iteration_", None))

    hp_outdir.mkdir(parents=True, exist_ok=True)
    with open(hp_outdir / f"{model_name}_fold{fold_k}.json", "w") as f:
        json.dump(best_params, f, indent=2)

    return best_params, y_hat

# -------------------- Main --------------------

def main():
    args = parse_args()
    outdir = Path(args.save_dir)
    (outdir / "oof").mkdir(parents=True, exist_ok=True)
    hp_outdir = outdir / "hp"

    df, Xnum, y, ids, folds_values, num_cols = load_and_merge(
        args.train, args.folds, args.id_col, args.target
    )
    unique_folds = sorted(np.unique(folds_values).tolist())

    # ===== XGB (API nativa) =====
    xgb_base = get_xgb_base_params(args.use_gpu, args.seed, args.xgb_n_estimators, args.xgb_learning_rate)
    xgb_oof = np.zeros_like(y, dtype=float)

    if args.tune_trials > 0:
        for k in unique_folds:
            _, y_hat = tune_one_fold(
                "xgb", xgb_base, Xnum, y, folds_values, k, num_cols,
                args.tune_trials, args.inner_splits, args.seed, args.use_gpu, hp_outdir,
                pruner_kind=args.pruner,
                asha_min_resource=args.asha_min_resource,
                asha_reduction=args.asha_reduction_factor,
                asha_min_esr=args.asha_min_early_stopping_rate,
            )
            xgb_oof[folds_values == k] = y_hat
    else:
        for k in unique_folds:
            y_hat = fit_predict_one_fold("xgb", xgb_base, Xnum, y, folds_values, k, num_cols, args.use_gpu, args.seed)
            xgb_oof[folds_values == k] = y_hat

    xgb_rmse, xgb_r2 = rmse(y, xgb_oof), float(r2(y, xgb_oof))
    oof_xgb_df = pd.DataFrame({
        "id": ids, "fold": folds_values, "y_true": y, "y_pred": xgb_oof, "model": "xgb", "seed": args.seed
    })
    oof_xgb_df.to_parquet(outdir / "oof" / "xgb.parquet", index=False)

    # ===== LGBM =====
    try:
        lgbm_oof = np.zeros_like(y, dtype=float)
        if args.tune_trials > 0:
            for k in unique_folds:
                try:
                    _, y_hat = tune_one_fold(
                        "lgbm", {}, Xnum, y, folds_values, k, num_cols,
                        args.tune_trials, args.inner_splits, args.seed, args.use_gpu, hp_outdir,
                        pruner_kind=args.pruner,
                        asha_min_resource=args.asha_min_resource,
                        asha_reduction=args.asha_reduction_factor,
                        asha_min_esr=args.asha_min_early_stopping_rate,
                    )
                except LightGBMError:
                    print("[WARN] LightGBM GPU falló en tuning; reintentando en CPU…")
                    _, y_hat = tune_one_fold(
                        "lgbm", {}, Xnum, y, folds_values, k, num_cols,
                        args.tune_trials, args.inner_splits, args.seed, False, hp_outdir,
                        pruner_kind=args.pruner,
                        asha_min_resource=args.asha_min_resource,
                        asha_reduction=args.asha_reduction_factor,
                        asha_min_esr=args.asha_min_early_stopping_rate,
                    )
                lgbm_oof[folds_values == k] = y_hat
        else:
            try:
                for k in unique_folds:
                    y_hat = fit_predict_one_fold("lgbm", {}, Xnum, y, folds_values, k, num_cols, args.use_gpu, args.seed)
                    lgbm_oof[folds_values == k] = y_hat
            except LightGBMError:
                print("[WARN] LightGBM GPU no disponible; reintentando en CPU…")
                for k in unique_folds:
                    y_hat = fit_predict_one_fold("lgbm", {}, Xnum, y, folds_values, k, num_cols, False, args.seed)
                    lgbm_oof[folds_values == k] = y_hat

        lgbm_rmse, lgbm_r2 = rmse(y, lgbm_oof), float(r2(y, lgbm_oof))
        oof_lgb_df = pd.DataFrame({
            "id": ids, "fold": folds_values, "y_true": y, "y_pred": lgbm_oof, "model": "lgbm", "seed": args.seed
        })
        oof_lgb_df.to_parquet(outdir / "oof" / "lgbm.parquet", index=False)
        metrics_lgb = {"rmse": lgbm_rmse, "r2": lgbm_r2}
    except Exception as e:
        print(f"[ERROR] LightGBM falló: {e}")
        oof_lgb_df = pd.DataFrame({
            "id": ids, "fold": folds_values, "y_true": y,
            "y_pred": np.full(len(y), np.nan), "model": "lgbm", "seed": args.seed
        })
        oof_lgb_df.to_parquet(outdir / "oof" / "lgbm.parquet", index=False)
        metrics_lgb = {"rmse": float("nan"), "r2": float("nan"), "error": str(e)}

    metrics = {
        "xgb":  {"rmse": xgb_rmse, "r2": xgb_r2},
        "lgbm": metrics_lgb,
        "n_samples": int(len(y)),
        "n_features_num": int(Xnum.shape[1]),
        "target": args.target,
        "used_gpu": args.use_gpu,
        "n_folds": int(len(np.unique(folds_values))),
        "tune_trials": args.tune_trials,
        "inner_splits": args.inner_splits,
        "pruner": args.pruner,
        "asha": {
            "min_resource": args.asha_min_resource,
            "reduction_factor": args.asha_reduction_factor,
            "min_early_stopping_rate": args.asha_min_early_stopping_rate
        }
    }
    with open(outdir / "metrics_tree.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] OOF guardado en:", (outdir / "oof").resolve())
    print(f"    XGB  → RMSE={metrics['xgb']['rmse']:.6f}  R2={metrics['xgb']['r2']:.6f}")
    print(f"    LGBM → RMSE={metrics['lgbm']['rmse']}  R2={metrics['lgbm']['r2']}")
if __name__ == "__main__":
    main()
