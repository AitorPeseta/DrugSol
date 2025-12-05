#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_infer_master.py (Pure Graph Strategy)
-------------------------------------------
Solo usa SMILES para Chemprop.
Ignora features extra para evitar errores de escala/orden.
"""
import argparse, json, sys, subprocess, numpy as np, pandas as pd, joblib, pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

def run_cmd(cmd, check=True):
    print(f"[CMD] {' '.join(map(str, cmd))}")
    try: subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e: print(f"[ERROR] {e}"); raise e

def read_any(path):
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)

def get_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return {}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "r2": float(r2_score(y_true[mask], y_pred[mask])),
        "n": int(mask.sum())
    }

def _norm_id(s): return s.astype("string").str.strip()

# --- PREDICT CHEMPROP (SIMPLIFICADO) ---
def predict_chemprop(model_dir, test_df, smiles_col, id_col, gpu=False):
    p = Path(model_dir)
    ckpts = sorted(list(p.rglob("*.pt")))
    if not ckpts: return pd.DataFrame({id_col: test_df[id_col]})
    ckpt = next((c for c in ckpts if c.name == "model.pt"), ckpts[0])
    
    print(f"[Infer] Chemprop v1 (Grafo Puro, ckpt={ckpt.name})...")
    
    tmp_in = p / "test_in.csv"
    tmp_out = p / "test_out.csv"
    
    df_in = test_df.copy()
    df_in = df_in.rename(columns={smiles_col: "smiles"})
    
    # Solo guardamos SMILES
    cols_to_save = ["smiles"]
    print(f"[Chemprop] Input cols: {cols_to_save}")
    df_in[cols_to_save].to_csv(tmp_in, index=False)
    
    cmd = [
        "chemprop_predict",
        "--test_path", str(tmp_in),
        "--preds_path", str(tmp_out),
        "--checkpoint_path", str(ckpt),
        "--batch_size", "64",
        "--smiles_columns", "smiles"
    ]
    
    if gpu: cmd += ["--gpu", "0"]
    
    try:
        run_cmd(cmd)
        preds = pd.read_csv(tmp_out)
        pred_col = [c for c in preds.columns if c != "smiles"][0]
        return pd.DataFrame({id_col: test_df[id_col].values, "y_chemprop": preds[pred_col].values})
    except Exception as e:
        print(f"[WARN] Chemprop inference failed: {e}")
        return pd.DataFrame({id_col: test_df[id_col]})

# --- PREDICT GBM (Igual) ---
def predict_gbm(models_dir, test_df, id_col):
    preds = pd.DataFrame({id_col: test_df[id_col]})
    xgb_path = Path(models_dir) / "xgb.pkl"
    if xgb_path.exists():
        try:
            print("[Infer] XGBoost...")
            xgb_pipe = joblib.load(xgb_path)
            X = test_df.select_dtypes(include=[np.number]).drop(columns=[id_col, "fold", "target", "logS"], errors='ignore')
            manifest = Path(models_dir) / "gbm_manifest.json"
            if manifest.exists():
                m = json.loads(manifest.read_text())
                for c in m.get("features", []): 
                    if c not in X.columns: X[c] = 0
                X = X[m.get("features", [])]
            preds["y_xgb"] = xgb_pipe.predict(X)
        except Exception as e: print(f"[WARN] XGB failed: {e}")

    lgbm_path = Path(models_dir) / "lgbm.pkl"
    if lgbm_path.exists():
        try:
            print("[Infer] LightGBM...")
            lgbm_pipe = joblib.load(lgbm_path)
            preds["y_lgbm"] = lgbm_pipe.predict(X)
        except Exception as e: print(f"[WARN] LGBM failed: {e}")
    return preds

def predict_tpsa(tpsa_json, test_df, id_col):
    if not tpsa_json or not Path(tpsa_json).exists(): return pd.DataFrame({id_col: test_df[id_col]})
    print("[Infer] Baseline...")
    try:
        model = json.loads(Path(tpsa_json).read_text())
        if "inv_temp" not in test_df.columns and "temp_C" in test_df.columns:
             test_df["inv_temp"] = 1000.0 / (test_df["temp_C"] + 273.15)
        
        y_pred = np.full(len(test_df), model.get("intercept", 0.0))
        for feat, w in model.get("coefs", {}).items():
            if feat in test_df.columns:
                y_pred += pd.to_numeric(test_df[feat], errors='coerce').fillna(0.0).values * w
        return pd.DataFrame({id_col: test_df[id_col], "y_tpsa": y_pred})
    except Exception as e:
        print(f"[WARN] TPSA failed: {e}")
        return pd.DataFrame({id_col: test_df[id_col]})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-tabular", required=True)
    ap.add_argument("--test-smiles", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--chemprop-model-dir", required=True)
    ap.add_argument("--tpsa-json", default=None)
    ap.add_argument("--save-dir", default="pred")
    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--weights-json", default=None)
    ap.add_argument("--stack-pkl", default=None)
    ap.add_argument("--chemprop-smiles-col", default=None)
    ap.add_argument("--tpsa-col", default="TPSA")
    ap.add_argument("--phenol-col", default="n_phenol")
    
    args = ap.parse_args()
    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)

    print("[Infer] Loading...")
    df_tab = read_any(args.test_tabular).drop_duplicates(subset=[args.id_col])
    df_smi = read_any(args.test_smiles).drop_duplicates(subset=[args.id_col])
    df_tab[args.id_col] = _norm_id(df_tab[args.id_col])
    df_smi[args.id_col] = _norm_id(df_smi[args.id_col])
    
    df_full = df_tab.merge(df_smi, on=args.id_col, suffixes=("", "_smi"))
    if "temp_C" in df_full.columns:
        t = pd.to_numeric(df_full["temp_C"], errors='coerce').fillna(25.0)
        df_full["inv_temp"] = 1000.0 / (t + 273.15)
    else: df_full["inv_temp"] = 3.35

    p_gbm = predict_gbm(args.models_dir, df_full, args.id_col)
    p_gnn = predict_chemprop(args.chemprop_model_dir, df_full, args.smiles_col, args.id_col, gpu=True)
    p_tpsa = predict_tpsa(args.tpsa_json, df_full, args.id_col)
    
    level0 = p_gbm.merge(p_gnn, on=args.id_col, how="left").merge(p_tpsa, on=args.id_col, how="left")
    
    for c in [args.target, "temp_C"]:
        if c in df_full.columns and c not in level0.columns:
            level0 = level0.merge(df_full[[args.id_col, c]], on=args.id_col, how="left")
            
    level0.to_parquet(outdir / "test_level0.parquet", index=False)
    
    cols = ["y_xgb", "y_lgbm", "y_chemprop", "y_tpsa"]
    present = [c for c in cols if c in level0.columns]
    X = level0[present].values
    
    if args.weights_json and Path(args.weights_json).exists():
        w = json.loads(Path(args.weights_json).read_text())
        vec = np.array([w.get(c.replace("y_", ""), 0.0) for c in present])
        if vec.sum() > 0: vec /= vec.sum()
        level0["y_pred_blend"] = np.nansum(X * vec, axis=1)
        level0[[args.id_col, "y_pred_blend"]].to_parquet(outdir / "test_blend.parquet", index=False)

    if args.stack_pkl and Path(args.stack_pkl).exists():
        try: meta = pickle.loads(Path(args.stack_pkl).read_bytes())
        except: meta = joblib.load(args.stack_pkl)
        if "coef" in meta:
            y_stk = np.full(len(level0), meta["intercept"])
            for f, w in zip(meta["feature_names"], meta["coef"]):
                if f in level0.columns: y_stk += level0[f].fillna(0.0).values * w
            level0["y_pred_stack"] = y_stk
            level0[[args.id_col, "y_pred_stack"]].to_parquet(outdir / "test_stack.parquet", index=False)

    if args.target in level0.columns:
        metrics = {}
        y_true = level0[args.target].values
        for c in present + ["y_pred_blend", "y_pred_stack"]:
            if c in level0.columns:
                metrics[c] = get_metrics(y_true, level0[c].values)
        if "temp_C" in level0.columns:
            t_val = pd.to_numeric(level0["temp_C"], errors='coerce')
            phys_mask = t_val.between(35, 38).values
            if phys_mask.any():
                metrics["physio_range"] = {}
                for c in present + ["y_pred_blend", "y_pred_stack"]:
                    if c in level0.columns:
                        metrics["physio_range"][c] = get_metrics(y_true[phys_mask], level0.loc[phys_mask, c].values)
        (outdir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()