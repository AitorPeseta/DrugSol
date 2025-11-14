#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 3: ENTRENAMIENTO FULL para XGB y LGBM (sin OOF).
- Carga TRAIN tabular, aplica preproc (imputer + variance threshold),
  entrena en TODO el TRAIN con mejores HP y guarda pipelines en models/.
- Si existe un directorio de HP por fold (p.ej. ./oof_gbm/hp/), agrega HP.
- Alternativamente, acepta ficheros --xgb-params y --lgbm-params (JSON).

Salidas:
  models/xgb.pkl
  models/lgbm.pkl
  models/gbm_manifest.json  (resumen de HP y columnas)
"""
import argparse, json, sys, math
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

import xgboost as xgb
import lightgbm as lgb
from lightgbm.basic import LightGBMError

RANDOM_STATE = 42

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser("Fase 3 · full train XGB+LGBM")
    ap.add_argument("--train", required=True, help="parquet/csv con features + target (+ id opcional)")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--id-col", default="row_uid")        
    ap.add_argument("--save-dir", default="models")
    ap.add_argument("--use-gpu", action="store_true")

    # Opcional: fracción valida para ES (0.0 = entreno full puro)
    ap.add_argument("--val-fraction", type=float, default=0.0)

    # HP sources
    ap.add_argument("--hp-dir", default=None, help="directorio con hp por fold (xgb_fold*.json, lgbm_fold*.json)")
    ap.add_argument("--xgb-params", default=None, help="JSON con mejores HP de XGB (opcional)")
    ap.add_argument("--lgbm-params", default=None, help="JSON con mejores HP de LGBM (opcional)")
    return ap.parse_args()

# ---------- utils ----------
def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).lower().endswith(".parquet") else pd.read_csv(path)

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

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

def agg_hp_from_dir(hp_dir: Path, prefix: str):
    """Agrega HP por fold (mediana para floats, modo para categ/ints)."""
    import statistics as st
    files = sorted(hp_dir.glob(f"{prefix}_fold*.json"))
    if not files: return None
    rows = []
    for f in files:
        try:
            rows.append(json.loads(Path(f).read_text()))
        except Exception:
            pass
    if not rows: return None
    keys = sorted({k for r in rows for k in r.keys()})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        try:
            nums = [float(v) for v in vals]
            out[k] = float(np.median(nums))
        except Exception:
            try:
                out[k] = st.mode(vals)
            except Exception:
                out[k] = vals[0]
    for k in ["max_depth","num_leaves","min_child_samples","bagging_freq","n_estimators","ffn_num_layers","depth"]:
        if k in out:
            out[k] = int(round(float(out[k])))
    return out

def load_hp(args):
    hp_xgb = None
    hp_lgb = None
    if args.hp_dir:
        hpdir = Path(args.hp_dir)
        if hpdir.exists():
            hp_xgb = agg_hp_from_dir(hpdir, "xgb")
            hp_lgb = agg_hp_from_dir(hpdir, "lgbm")
    if args.xgb_params:
        hp_xgb = json.loads(Path(args.xgb_params).read_text())
    if args.lgbm_params:
        hp_lgb = json.loads(Path(args.lgbm_params).read_text())
    return hp_xgb, hp_lgb

def get_xgb(hp: dict | None, use_gpu: bool):
    params = dict(
        n_estimators=2000, learning_rate=0.03,
        max_depth=8, min_child_weight=2.0,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        tree_method="gpu_hist" if use_gpu else "hist",
    )
    if hp:
        params.update({k: v for k, v in hp.items()
                       if k in {"n_estimators","learning_rate","max_depth","min_child_weight",
                                "subsample","colsample_bytree","reg_lambda","reg_alpha"}})
    return xgb.XGBRegressor(**params)

def get_lgbm(hp: dict | None, use_gpu: bool):
    params = dict(
        n_estimators=3000, learning_rate=0.03,
        num_leaves=256, max_depth=-1, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.5,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        objective="regression", metric="rmse",
        device_type="gpu" if use_gpu else "cpu"
    )
    if hp:
        ren = {"lambda_l1":"reg_alpha", "lambda_l2":"reg_lambda",
               "feature_fraction":"colsample_bytree", "bagging_fraction":"subsample"}
        hp2 = hp.copy()
        for a,b in ren.items():
            if a in hp2 and b not in hp2: hp2[b]=hp2[a]
        params.update({k:v for k,v in hp2.items()
                       if k in {"n_estimators","learning_rate","num_leaves","max_depth",
                                "min_child_samples","subsample","colsample_bytree",
                                "reg_alpha","reg_lambda"}})
    return lgb.LGBMRegressor(**params)

# ---------- main ----------
def main():
    args = parse_args()
    outdir = Path(getattr(args, "save_dir", "models"))
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_any(args.train)
    if args.target not in df.columns:
        raise SystemExit(f"Falta target '{args.target}' en TRAIN")

    # --- Selección de features (defensiva) ---
    to_exclude = {
        args.target,                # el target nunca es feature
        args.id_col,                # id
        "fold",                     # por si viniera del OOF
        "smiles_neutral", "smiles_original", "smiles"  # textos SMILES
    }
    num_all = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in to_exclude if c in num_all.columns]
    if drop_cols:
        num_all.drop(columns=drop_cols, inplace=True)

    if isinstance(num_all, pd.Series):
        num_all = num_all.to_frame()
    num_cols = num_all.columns.tolist()
    if not num_cols:
        raise SystemExit("No hay columnas numéricas válidas tras filtrar (¿quedó solo target/id/fold?).")

    y = df[args.target].to_numpy()
    pre = make_preproc(num_cols)

    hp_xgb, hp_lgb = load_hp(args)

    # --- split opcional para ES ---
    use_val = False
    if args.val_fraction and 0.0 < float(args.val_fraction) < 0.5:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.arange(len(df)); rng.shuffle(idx)
        n_val = int(round(len(idx) * float(args.val_fraction)))
        n_val = max(1, min(n_val, len(idx) - 2))
        va_idx = idx[:n_val]; tr_idx = idx[n_val:]

        X_tr = pre.fit_transform(num_all.iloc[tr_idx]); y_tr = y[tr_idx]
        X_va = pre.transform(num_all.iloc[va_idx]);     y_va = y[va_idx]
        _ = pre.fit(num_all)  # re-ajusta en TODO para entreno final
        use_val = True
    else:
        _ = pre.fit(num_all)
        X_tr = X_va = y_tr = y_va = None, None, None, None
        use_val = False

    # ---- XGB ----
    xgb_est = get_xgb(hp_xgb, args.use_gpu)
    if use_val:
        try:
            xgb_est.set_params(early_stopping_rounds=100)
            xgb_est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse", verbose=False)
            best_iter = getattr(xgb_est, "best_iteration", None)
            if best_iter is not None:
                xgb_est.set_params(n_estimators=int(best_iter + 1))
        except TypeError:
            pass
    xgb_pipe = Pipeline([
        ("pre", pre),
        ("m", get_xgb({**(hp_xgb or {}), "n_estimators": xgb_est.get_params().get("n_estimators", 2000)},
                      args.use_gpu))
    ])
    xgb_pipe.fit(num_all, y)

    # ---- LGBM ----
    try:
        lgbm_est = get_lgbm(hp_lgb, args.use_gpu)
        if use_val:
            lgbm_est.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            best_it = getattr(lgbm_est, "best_iteration_", None)
            if best_it is not None:
                lgbm_est.set_params(n_estimators=int(best_it))
        lgbm_pipe = Pipeline([("pre", pre), ("m", lgbm_est)])
        lgbm_pipe.fit(num_all, y)
    except LightGBMError:
        print("[WARN] LGBM GPU no disponible; reintentando en CPU…", file=sys.stderr)
        lgbm_est = get_lgbm(hp_lgb, use_gpu=False)
        lgbm_pipe = Pipeline([("pre", pre), ("m", lgbm_est)])
        lgbm_pipe.fit(num_all, y)

    # ---- guardar ----
    import pickle
    (outdir / "xgb.pkl").write_bytes(pickle.dumps(xgb_pipe))
    (outdir / "lgbm.pkl").write_bytes(pickle.dumps(lgbm_pipe))

    manifest = {
        "target": args.target,
        "n_samples": int(len(y)),
        "n_features_used": int(len(num_cols)),
        "id_col": args.id_col,
        "val_fraction": float(args.val_fraction),
        "xgb_params": xgb_pipe.named_steps["m"].get_params(),
        "lgbm_params": lgbm_pipe.named_steps["m"].get_params(),
        "ignored_text_cols": ["smiles_neutral", "smiles_original", "smiles", "fold", args.id_col]
    }
    clean = sanitize(manifest)
    with open(outdir / "gbm_manifest.json", "w") as f:
        json.dump(clean, f, indent=2)

    print("[OK] Guardados modelos full en:", outdir.resolve())
    print(f"   · n_features usadas: {len(num_cols)}")

if __name__ == "__main__":
    main()
