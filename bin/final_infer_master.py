#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


# ============================= utilidades base ================================

def _norm_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _norm_smiles(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _must_have(df: pd.DataFrame, col: str, name: str):
    if col not in df.columns:
        raise ValueError(f"[FATAL] {name} no tiene la columna obligatoria '{col}'")


def _must_be_unique(df: pd.DataFrame, col: str, name: str):
    _must_have(df, col, name)
    if df[col].isna().any():
        na = int(df[col].isna().sum())
        raise ValueError(f"[FATAL] {name}.{col} tiene {na} NAs")
    if not df[col].is_unique:
        dups = int(df[col].size - df[col][~df[col].duplicated()].size)
        raise ValueError(f"[FATAL] {name}.{col} no es único (dups={dups})")


def _dump_ids(path: Path, ids: pd.Series, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_uid": _norm_id(ids)}).to_csv(path, index=False)
    print(f"[DEBUG] {title} guardado en {path.resolve()}")


def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[OK] Guardado Parquet: {path.resolve()}")


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Formato no soportado: {path.suffix} ({path})")


# ====================== helpers columnas/features/GBM =========================

def _extract_feature_names_in(mdl) -> Optional[List[str]]:
    """Intenta extraer la lista exacta de columnas usadas en train."""
    try:
        if hasattr(mdl, "feature_names_in_"):
            return list(mdl.feature_names_in_)
        from sklearn.pipeline import Pipeline
        if isinstance(mdl, Pipeline):
            if hasattr(mdl, "feature_names_in_"):
                return list(mdl.feature_names_in_)
            for _, step in mdl.named_steps.items():
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
            last = mdl.steps[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
    except Exception:
        pass
    return None


def _load_manifest_feature_cols(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        j = json.loads(path.read_text())
        for k in ["feature_cols", "features", "feature_columns"]:
            v = j.get(k)
            if isinstance(v, list):
                return list(v)
    except Exception:
        pass
    return None


def _reorder_X_to_feature_list(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"[FATAL] Faltan columnas requeridas por el modelo GBM: "
            f"{missing[:12]}{'...' if len(missing) > 12 else ''}"
        )
    return df[features].copy()


def _prep_X_for_model(df: pd.DataFrame, id_col: str, target: Optional[str],
                      manifest_cols: Optional[List[str]], mdl) -> pd.DataFrame:
    base = df.copy()
    drop = [id_col]
    if target and target in base.columns:
        drop.append(target)
    for s in ["smiles", "smiles_neutral", "SMILES", "Smiles"]:
        if s in base.columns:
            drop.append(s)
    base = base.drop(columns=list(set(drop) & set(base.columns)))
    names = _extract_feature_names_in(mdl) or manifest_cols
    if not names:
        raise ValueError(
            "[FATAL] No se pudieron determinar las columnas exactas para los GBM.\n"
            "Proporciona gbm_manifest.json con 'feature_cols' o guarda los PKL con feature_names_in_."
        )
    return _reorder_X_to_feature_list(base, names)


# ===================== helpers Chemprop / SMILES / dups =======================

def _apply_stack_object(stack_obj, level0_out: pd.DataFrame, pred_cols: list) -> np.ndarray:
    import numpy as _np
    X_meta_default = level0_out[pred_cols].to_numpy(dtype=float)

    # Caso 1: estimador sklearn directo (Ridge, Pipeline, etc.)
    if hasattr(stack_obj, "predict") and callable(stack_obj.predict):
        return _np.asarray(stack_obj.predict(X_meta_default)).ravel()

    # Caso 2: dict serializado (formatos varios)
    if isinstance(stack_obj, dict):
        d = stack_obj

        # Si viene como {"labels": [...], "model": Ridge}
        mdl = d.get("model", None)
        if mdl is not None and hasattr(mdl, "predict") and callable(mdl.predict):
            return _np.asarray(mdl.predict(X_meta_default)).ravel()

        # Formato: coef / intercept / feature_names
        coef = d.get("coef", d.get("coef_", None))
        if coef is not None:
            coef = _np.asarray(coef, dtype=float).ravel()
            intercept = float(d.get("intercept", d.get("intercept_", 0.0)))
            feat_names = d.get("feature_names", d.get("features", None))

            if feat_names is not None:
                feat_names = list(feat_names)
                present = [c for c in feat_names if c in level0_out.columns]
                missing = [c for c in feat_names if c not in level0_out.columns]

                if missing:
                    print(f"[WARN] stack_pkl pide columnas {missing} que no están en level0; se ignorarán.")

                if not present:
                    raise ValueError("[FATAL] Ninguna de las columnas requeridas por stack_pkl está en level0.")

                if len(coef) != len(feat_names):
                    raise ValueError(
                        f"[FATAL] len(coef)={len(coef)} != len(feature_names)={len(feat_names)}"
                    )

                coef_map = {name: coef[i] for i, name in enumerate(feat_names)}
                coef_vec = _np.array([coef_map[c] for c in present], dtype=float)
                X_meta = level0_out[present].to_numpy(dtype=float)
            else:
                if len(coef) != len(pred_cols):
                    raise ValueError(
                        f"[FATAL] len(coef)={len(coef)} != len(pred_cols)={len(pred_cols)} ({pred_cols})"
                    )
                X_meta = X_meta_default
                coef_vec = coef

            return X_meta.dot(coef_vec) + intercept

        # Otros formatos de pesos simples
        weights = d.get("weights", None)
        if isinstance(weights, dict):
            alias = {
                "xgb": "y_xgb", "lgbm": "y_lgbm",
                "gnn": "y_chemprop", "chemprop": "y_chemprop", "tpsa": "y_tpsa"
            }
            w_by_col = {}
            for k, v in weights.items():
                col = alias.get(k, k)
                w_by_col[col] = float(v)
            wvec = _np.array([float(w_by_col.get(c, 0.0)) for c in pred_cols], dtype=float)
            return X_meta_default.dot(wvec) + float(d.get("intercept", 0.0))

        def _extract_weights_full(dd):
            wf = None
            if isinstance(dd.get("blend"), dict) and isinstance(dd["blend"].get("weights_full"), dict):
                wf = dd["blend"]["weights_full"]
            elif isinstance(dd.get("weights_full"), dict):
                wf = dd["weights_full"]
            return wf

        wf = _extract_weights_full(d)
        if isinstance(wf, dict):
            alias = {
                "xgb": "y_xgb", "lgbm": "y_lgbm",
                "gnn": "y_chemprop", "chemprop": "y_chemprop", "tpsa": "y_tpsa",
                "y_xgb": "y_xgb", "y_lgbm": "y_lgbm",
                "y_chemprop": "y_chemprop", "y_tpsa": "y_tpsa"
            }
            w_by_col = {alias.get(k, k): float(v) for k, v in wf.items()}
            wvec = _np.array([float(w_by_col.get(c, 0.0)) for c in pred_cols], dtype=float)
            return X_meta_default.dot(wvec)

        labels = d.get("labels", None)
        w_vec = d.get("weights", None)
        if isinstance(labels, (list, tuple)) and isinstance(w_vec, (list, tuple)):
            alias = {
                "xgb": "y_xgb", "lgbm": "y_lgbm",
                "gnn": "y_chemprop", "chemprop": "y_chemprop", "tpsa": "y_tpsa",
                "y_xgb": "y_xgb", "y_lgbm": "y_lgbm",
                "y_chemprop": "y_chemprop", "y_tpsa": "y_tpsa"
            }
            label_to_w = {}
            for lab, w in zip(labels, w_vec):
                col = alias.get(str(lab), str(lab))
                label_to_w[col] = float(w)
            wvec = _np.array([float(label_to_w.get(c, 0.0)) for c in pred_cols], dtype=float)
            return X_meta_default.dot(wvec) + float(d.get("intercept", 0.0))

        raise ValueError("[FATAL] --stack-pkl dict no coincide con formatos soportados.")

    raise TypeError("[FATAL] --stack-pkl debe ser un estimator sklearn o un dict soportado.")


def _pick_smiles_col(df: pd.DataFrame, preferred: Optional[str], default_hint: str) -> str:
    candidates = [preferred, default_hint, "smiles", "smiles_neutral", "SMILES", "Smiles"]
    for c in candidates:
        if c and c in df.columns:
            return c
    raise KeyError(
        "No se encontró ninguna columna de SMILES. "
        f"Probadas: {candidates}. Disponibles: {list(df.columns)}"
    )


def _smiles_cumkey(df: pd.DataFrame, smiles_col: str, key_name: str = "_cum") -> pd.DataFrame:
    out = df.copy()
    out[key_name] = out.groupby(smiles_col).cumcount()
    return out


def _find_pred_col(df_or_path, target: Optional[str] = None) -> str:
    import pandas as _pd
    from pathlib import Path as _Path
    if isinstance(df_or_path, (str, _Path)):
        p = _Path(df_or_path)
        df = _pd.read_parquet(p) if p.suffix.lower() == ".parquet" else _pd.read_csv(p)
    else:
        df = df_or_path
    if target and target in df.columns:
        return target
    lower = {c.lower(): c for c in df.columns}
    for key in ["y_chemprop", "prediction", "pred", "value", "target", "y_hat", "logS"]:
        if key in lower:
            return lower[key]
    if "smiles" in df.columns and len(df.columns) == 2:
        other = [c for c in df.columns if c != "smiles"]
        if other:
            return other[0]
    raise ValueError(f"No se encontró columna de predicción en {df.columns.tolist()}")


def _chemprop_models(chem_dir: Path) -> List[Path]:
    paths: List[Path] = []
    m0 = chem_dir / "model_0"
    if (m0 / "best.pt").exists():
        paths.append(m0 / "best.pt")
    ck = m0 / "checkpoints"
    if ck.exists():
        paths += sorted(ck.glob("*.ckpt"))
    if not paths:
        paths = sorted(list(chem_dir.rglob("*.pt")) + list(chem_dir.rglob("*.ckpt")))
    if not paths:
        raise FileNotFoundError(f"No hay checkpoints .pt/.ckpt en {chem_dir}")
    return paths


def _chemprop_predict_legacy(model_paths: List[Path], test_csv: Path, preds_csv: Path):
    """
    Invocación “vieja”: usamos SOLO SMILES, sin descriptores extra.
    Debe coincidir con cómo se entrenó el modelo GNN (como en tu script original).
    """
    present_cols = ["temp_C","n_ionizable","n_acid","n_base",
                                "TPSA","logP","HBD","HBA","FractionCSP3","MW"]
    cmd = ["chemprop", "predict", "-i", str(test_csv), "-o", str(preds_csv), "--model-paths"]
    cmd += ["--descriptors-columns", *present_cols]
    cmd += [str(p) for p in model_paths]
    print("[INFO] Ejecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ============================== blend / stack =================================

def _blend_with_rowwise_renorm(mat: np.ndarray, w: np.ndarray) -> np.ndarray:
    W = np.tile(w, (mat.shape[0], 1))
    mask = ~np.isnan(mat)
    W_eff = W * mask
    den = W_eff.sum(axis=1)
    den[den == 0] = np.nan
    return np.nansum(mat * W, axis=1) / den


def _blend_weights_from_json(weights_json: Path, pred_cols: List[str]) -> np.ndarray:
    w_json = json.loads(weights_json.read_text())
    keymap = {"y_xgb": "xgb", "y_lgbm": "lgbm", "y_chemprop": "chemprop", "y_tpsa": "tpsa"}
    alias_map = {"chemprop": "chemprop", "gnn": "chemprop", "xgb": "xgb", "lgbm": "lgbm", "tpsa": "tpsa"}

    def _get_weight_for(col: str) -> float:
        key = keymap.get(col, col)
        if key in w_json:
            return float(w_json[key])
        for k, v in alias_map.items():
            if v == key and k in w_json:
                return float(w_json[k])
        if col in w_json:
            return float(w_json[col])
        return 0.0

    w_raw = np.array([_get_weight_for(c) for c in pred_cols], dtype=float)
    if w_raw.sum() > 0:
        w = w_raw / w_raw.sum()
    else:
        w = np.ones(len(pred_cols), dtype=float) / max(1, len(pred_cols))
    return w


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "r2": float(r2), "n": int(len(y_true))}


# ======================= GBM / TPSA: cargar o generar preds ===================

def _maybe_make_gbm_preds_from_pkls(models_dir: Path,
                                    test_tab: pd.DataFrame,
                                    id_col: str,
                                    target: Optional[str],
                                    save_dir: Path) -> pd.DataFrame:
    out = test_tab[[id_col]].copy()
    out[id_col] = _norm_id(out[id_col])

    xgb_parq = next((p for p in [models_dir / "pred" / "xgb_test.parquet",
                                 models_dir / "xgb_test.parquet",
                                 save_dir / "xgb_test.parquet"] if p.exists()), None)
    lgbm_parq = next((p for p in [models_dir / "pred" / "lgbm_test.parquet",
                                  models_dir / "lgbm_test.parquet",
                                  save_dir / "lgbm_test.parquet"] if p.exists()), None)

    if xgb_parq:
        xdf = pd.read_parquet(xgb_parq)
        if id_col in xdf.columns and "y_xgb" in xdf.columns:
            xdf[id_col] = _norm_id(xdf[id_col])
            out = out.merge(xdf[[id_col, "y_xgb"]].drop_duplicates(id_col), on=id_col, how="left")
            print(f"[INFO] Cargado XGB preds: {xgb_parq.resolve()}")

    if lgbm_parq:
        ldf = pd.read_parquet(lgbm_parq)
        if id_col in ldf.columns and "y_lgbm" in ldf.columns:
            ldf[id_col] = _norm_id(ldf[id_col])
            out = out.merge(ldf[[id_col, "y_lgbm"]].drop_duplicates(id_col), on=id_col, how="left")
            print(f"[INFO] Cargado LGBM preds: {lgbm_parq.resolve()}")

    if ("y_xgb" in out.columns and out["y_xgb"].notna().any()) or ("y_lgbm" in out.columns and out["y_lgbm"].notna().any()):
        return out

    xgb_pkl = models_dir / "xgb.pkl"
    lgbm_pkl = models_dir / "lgbm.pkl"
    if not xgb_pkl.exists() and not lgbm_pkl.exists():
        return out

    import joblib
    manifest_cols = _load_manifest_feature_cols(models_dir / "gbm_manifest.json")

    if xgb_pkl.exists():
        try:
            xgb = joblib.load(xgb_pkl)
            X = _prep_X_for_model(test_tab, id_col, target, manifest_cols, xgb)
            out["y_xgb"] = np.asarray(xgb.predict(X)).ravel()
            _save_parquet(out[[id_col, "y_xgb"]], save_dir / "xgb_test.parquet")
            print("[INFO] Generado y guardado XGB preds desde PKL (con columnas alineadas)")
        except Exception as e:
            print(f"[WARN] No se pudo predecir con xgb.pkl: {e}")

    if lgbm_pkl.exists():
        try:
            lgb = joblib.load(lgbm_pkl)
            X = _prep_X_for_model(test_tab, id_col, target, manifest_cols, lgb)
            out["y_lgbm"] = np.asarray(lgb.predict(X)).ravel()
            _save_parquet(out[[id_col, "y_lgbm"]], save_dir / "lgbm_test.parquet")
            print("[INFO] Generado y guardado LGBM preds desde PKL (con columnas alineadas)")
        except Exception as e:
            print(f"[WARN] No se pudo predecir con lgbm.pkl: {e}")

    return out


def _maybe_add_tpsa_pred(out_df: pd.DataFrame,
                         test_tab: pd.DataFrame,
                         id_col: str,
                         tpsa_json: Optional[Path],
                         tpsa_col: str,
                         phenol_col: Optional[str]) -> pd.DataFrame:
    """
    Añade y_tpsa si se proporciona un modelo TPSA.

    Soporta dos formatos de JSON:

    1) **Nuevo (full Ridge)** (salida de train_full_tpsa_mp.py, por ejemplo)
       {
         "intercept": ...,
         "coef": {"TPSA": w1, "logP": w2, "mp_m25": w3, "aroOHdel": w4, ...},
         "features": ["TPSA","logP","mp_m25","aroOHdel",...],
       }
       y_tpsa = intercept + sum_i coef[feat_i] * x[feat_i]

       - Si falta algún feature pero se puede derivar:
           * "mp_m25"  -> a partir de columna "mp" (ºC): mp_m25 = mp - 25
           * "_sqrt_*" -> a partir de la columna base: sqrt( max(col, 0) )
           * "aroOHdel" -> n_phenol + n_phenol_like (si existen)
         se crea en test_tab al vuelo.
       - Si después de eso sigue faltando algo, se emite WARN y no se añade y_tpsa.

    2) **Antiguo** (lineal simple):
       {"a":..., "b":..., "c": opcional}
       y_tpsa = a * TPSA + b (+ c * phenol_col)
    """
    if tpsa_json is None:
        return out_df

    if not tpsa_json.exists():
        print(f"[WARN] --tpsa-json {tpsa_json} no existe; se ignora modelo TPSA.")
        return out_df

    try:
        cfg = json.loads(tpsa_json.read_text())
    except Exception as e:
        print(f"[WARN] No se pudo leer JSON de {tpsa_json}: {e}")
        return out_df

    # --- Formato nuevo: con 'coef' ---
    if isinstance(cfg, dict) and "coef" in cfg:
        coef_map = cfg.get("coef", {})
        if not isinstance(coef_map, dict) or not coef_map:
            print(f"[WARN] tpsa_json {tpsa_json} tiene 'coef' vacío; se ignora.")
            return out_df

        intercept = float(cfg.get("intercept", 0.0))
        features = cfg.get("features")
        if not isinstance(features, list) or not features:
            features = list(coef_map.keys())

        tab = test_tab.copy()

        # Intentar crear columnas derivadas (mp_m25, _sqrt_*, aroOHdel)
        for feat in features:
            if feat in tab.columns:
                continue

            # mp_m25 a partir de 'mp'
            if feat == "mp_m25" and "mp" in tab.columns:
                tab["mp_m25"] = pd.to_numeric(tab["mp"], errors="coerce") - 25.0
                continue

            # _sqrt_X a partir de X (p.ej. _sqrt_TPSA)
            if feat.startswith("_sqrt_"):
                base_col = feat[len("_sqrt_"):]
                if base_col in tab.columns:
                    tab[feat] = np.sqrt(
                        np.clip(pd.to_numeric(tab[base_col], errors="coerce"), a_min=0, a_max=None)
                    )
                    continue

            # aroOHdel = n_phenol + n_phenol_like
            if feat == "aroOHdel":
                have_nphen = "n_phenol" in tab.columns
                have_nlike = "n_phenol_like" in tab.columns
                if have_nphen or have_nlike:
                    nphen = pd.to_numeric(tab["n_phenol"], errors="coerce") if have_nphen else 0
                    nlike = pd.to_numeric(tab["n_phenol_like"], errors="coerce") if have_nlike else 0
                    tab["aroOHdel"] = (
                        (nphen if hasattr(nphen, "__array__") else 0)
                        + (nlike if hasattr(nlike, "__array__") else 0)
                    )
                    continue

        # Comprobamos que ya están todas las features
        missing = [f for f in features if f not in tab.columns]
        if missing:
            print(f"[WARN] No se puede aplicar modelo TPSA (faltan columnas {missing}); se ignora y_tpsa.")
            return out_df

        # Valores numéricos y predicción
        X_cols = []
        for f in features:
            X_cols.append(pd.to_numeric(tab[f], errors="coerce").values)
        X = np.vstack(X_cols).T  # shape (n, n_features)
        coefs = np.array([float(coef_map[f]) for f in features], dtype=float)
        y_vals = intercept + X.dot(coefs)

        tdf = pd.DataFrame({
            id_col: _norm_id(tab[id_col]) if id_col in tab.columns else _norm_id(test_tab[id_col]),
            "y_tpsa": y_vals
        })
        return out_df.merge(tdf[[id_col, "y_tpsa"]], on=id_col, how="left")

    # --- Formato antiguo: a,b,c ---
    try:
        a = float(cfg["a"])
        b = float(cfg["b"])
        c = float(cfg.get("c", 0.0))
    except Exception as e:
        print(f"[WARN] tpsa_json no tiene ni 'coef' ni (a,b,c) válidos: {e}")
        return out_df

    if tpsa_col not in test_tab.columns:
        print(f"[WARN] No está la columna TPSA '{tpsa_col}' en test_tab; se ignora TPSA.")
        return out_df

    df = test_tab[[id_col, tpsa_col]].copy()
    df[id_col] = _norm_id(df[id_col])
    x = pd.to_numeric(df[tpsa_col], errors="coerce")

    extra = 0.0
    if phenol_col and phenol_col in test_tab.columns and c != 0.0:
        ph = pd.to_numeric(test_tab[phenol_col], errors="coerce").fillna(0.0)
        extra = c * ph.values
    else:
        extra = 0.0

    y = a * x.values + b + extra
    tdf = pd.DataFrame({id_col: df[id_col].values, "y_tpsa": y})
    return out_df.merge(tdf, on=id_col, how="left")


# ================================== main =====================================

def main():
    ap = argparse.ArgumentParser("final_infer_master")
    ap.add_argument("--test-tabular", required=True, type=str)
    ap.add_argument("--test-smiles", required=True, type=str)
    ap.add_argument("--models-dir", default="models", type=str)
    ap.add_argument("--chemprop-model-dir", required=True, type=str)
    ap.add_argument("--save-dir", default="pred", type=str)

    ap.add_argument("--id-col", default="row_uid")
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--chemprop-smiles-col", default=None)
    ap.add_argument("--target", default=None)

    ap.add_argument("--weights-json", default=None, type=str)
    ap.add_argument("--stack-pkl", default=None, type=str)
    ap.add_argument("--chemprop-preds", default=None, type=str,
                    help="CSV/Parquet con predicciones de chemprop. Columnas admitidas: "
                         "(row_uid,y_chemprop) o (smiles,y_chemprop) con el mismo orden/duplicados que test_smiles.")

    # --- TPSA (+ phenol + mp) ---
    ap.add_argument("--tpsa-json", default=None, type=str,
                    help="JSON del modelo TPSA. "
                         "Formato nuevo: {'intercept':..,'coef':{'TPSA':..,'logP':..,'mp_m25':..,'aroOHdel':..},'features':[...]}. "
                         "Formato antiguo (legacy): {'a':..,'b':..,'c': opcional}.")
    ap.add_argument("--tpsa-col", default="TPSA", type=str,
                    help="Nombre de columna TPSA en test tabular (se usa sobre todo en formato antiguo).")
    ap.add_argument("--phenol-col", default="phenol_count", type=str,
                    help="Descriptor de fenoles (sólo usado si el JSON es antiguo a,b,c).")

    args = ap.parse_args()

    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------- carga y normalización de TEST -----------------------
    test_tab = _read_parquet_or_csv(Path(args.test_tabular)).copy()
    test_smi = _read_parquet_or_csv(Path(args.test_smiles)).copy()

    _must_have(test_tab, args.id_col, "test_tab")
    test_tab[args.id_col] = _norm_id(test_tab[args.id_col])
    _must_be_unique(test_tab, args.id_col, "test_tab")

    _must_have(
        test_smi,
        args.smiles_col if args.chemprop_smiles_col is None
        else _pick_smiles_col(test_smi, args.chemprop_smiles_col, args.smiles_col),
        "test_smi (smiles)"
    )
    if args.id_col in test_smi.columns:
        test_smi[args.id_col] = _norm_id(test_smi[args.id_col])

    print(f"[DEBUG] test_tab rows={len(test_tab)}, test_smi rows={len(test_smi)}")

    # -------------------- base predictions (GBM) ------------------------------
    base_tab = _maybe_make_gbm_preds_from_pkls(
        Path(args.models_dir),
        test_tab,
        args.id_col,
        args.target,
        Path(args.save_dir)
    )
    _must_be_unique(base_tab, args.id_col, "base_tab")
    print(f"[DEBUG] base_tab rows={len(base_tab)} (ids únicos)")

    # ==================== CHEMPROP (primero) ==================================
    if args.chemprop_preds:
        cp_raw = _read_parquet_or_csv(Path(args.chemprop_preds)).copy()
        cols_lower = {c.lower(): c for c in cp_raw.columns}
        if args.id_col in cp_raw.columns:
            pred_col = _find_pred_col(cp_raw.drop(columns=[args.id_col], errors="ignore"), target=args.target)
            cp_preds = cp_raw[[args.id_col, pred_col]].rename(columns={pred_col: "y_chemprop"}).copy()
        elif "smiles" in cols_lower:
            pred_col = _find_pred_col(cp_raw, target=args.target)
            test_key = _smiles_cumkey(
                test_smi[[args.id_col, args.smiles_col]].copy(),
                args.smiles_col,
                "_cum"
            ).rename(columns={args.smiles_col: "smiles"})
            cp_key = _smiles_cumkey(
                cp_raw[["smiles", pred_col]].rename(columns={"smiles": "smiles"}),
                "smiles",
                "_cum"
            )
            tmp = test_key.merge(cp_key.rename(columns={pred_col: "y_chemprop"}),
                                 on=["smiles", "_cum"], how="left")
            cp_preds = tmp[[args.id_col, "y_chemprop"]].copy()
        else:
            raise ValueError(
                "Formato de --chemprop-preds no reconocido. "
                "Esperado (row_uid,y_chemprop) o (smiles,y_chemprop)."
            )

        cp_preds[args.id_col] = _norm_id(cp_preds[args.id_col])
        _must_be_unique(cp_preds, args.id_col, "cp_preds (from file)")
        print(f"[INFO] Usando predicciones de Chemprop desde archivo: {args.chemprop_preds}")

    else:
        chem_dir = Path(args.chemprop_model_dir)
        model_paths = _chemprop_models(chem_dir)

        chem_col = _pick_smiles_col(test_smi, args.chemprop_smiles_col, args.smiles_col)

        # Sólo pasamos SMILES a chemprop; mismo esquema que en entrenamiento
        tmp_rows = test_smi[[chem_col]].copy()
        tmp_rows[chem_col] = _norm_smiles(tmp_rows[chem_col])
        tmp_keyed = _smiles_cumkey(tmp_rows, chem_col, "_cum").rename(columns={chem_col: "smiles"})

        tmp_test_csv = Path(args.save_dir) / "chemprop_test_input.csv"
        tmp_keyed[["smiles"]].to_csv(tmp_test_csv, index=False)

        preds_path = Path(args.save_dir) / "chemprop_test_rows.csv"
        _chemprop_predict_legacy(model_paths, tmp_test_csv, preds_path)

        preds_df = pd.read_csv(preds_path)
        pred_col = _find_pred_col(preds_df, target=args.target)
        if "smiles" not in preds_df.columns:
            preds_df["smiles"] = pd.read_csv(tmp_test_csv)["smiles"].values
        preds_df["smiles"] = _norm_smiles(preds_df["smiles"])
        preds_df = preds_df.rename(columns={pred_col: "y_chemprop"})[["smiles", "y_chemprop"]]

        preds_keyed = _smiles_cumkey(preds_df, "smiles", "_cum")
        test_key = tmp_keyed
        cp_preds = test_key.merge(preds_keyed, on=["smiles", "_cum"], how="left") \
                           .drop(columns=["smiles", "_cum"])
        cp_preds[args.id_col] = _norm_id(test_smi[args.id_col]) if args.id_col in test_smi.columns else pd.Series(np.arange(len(cp_preds))).astype("string")
        _must_have(cp_preds, "y_chemprop", "cp_preds")
        cp_preds = pd.concat([test_smi[[args.id_col]].reset_index(drop=True), cp_preds[["y_chemprop"]]], axis=1)
        _must_be_unique(cp_preds, args.id_col, "cp_preds")

    print(f"[DEBUG] cp_preds rows={len(cp_preds)} (ids únicos)")

    # ==================== TPSA (después de chemprop) ==========================
    tpsa_json = Path(args.tpsa_json) if args.tpsa_json else None

    # Mezclamos features de test_smiles en test_tab para TPSA
    features_df = test_tab.copy()
    if args.id_col in test_smi.columns:
        cols_tpsa = [
            c for c in [
                "logP", "TPSA", "mp",
                "n_phenol", "n_phenol_like",
                "phenol_like", "has_phenol"
            ]
            if c in test_smi.columns
        ]
        if cols_tpsa:
            aux = test_smi[[args.id_col] + cols_tpsa].drop_duplicates(args.id_col)
            aux[args.id_col] = _norm_id(aux[args.id_col])
            features_df = features_df.merge(aux, on=args.id_col, how="left")
            print(f"[INFO] Enriquecido test_tab con columnas TPSA/logP/mp/phenol desde test_smiles: {cols_tpsa}")

    base_tab = _maybe_add_tpsa_pred(
        out_df=base_tab,
        test_tab=features_df,
        id_col=args.id_col,
        tpsa_json=tpsa_json,
        tpsa_col=args.tpsa_col,
        phenol_col=args.phenol_col
    )

    # -------------------- unión segura y level0 --------------------------------
    union_df = base_tab.merge(
        cp_preds.drop_duplicates(subset=[args.id_col]),
        on=args.id_col, how="outer"
    ).drop_duplicates(subset=[args.id_col])

    level0_cols = [args.id_col]
    for c in ["y_xgb", "y_lgbm", "y_chemprop", "y_tpsa"]:
        if c in union_df.columns:
            level0_cols.append(c)
    level0_out = union_df[level0_cols].copy()

    # % NA en y_chemprop tras la unión (por ids)
    if "y_chemprop" in level0_out.columns:
        na_rate = float(level0_out["y_chemprop"].isna().mean())
        print(f"[DEBUG] y_chemprop NA rate after union: {na_rate:.3f}")
        if na_rate > 0.10:
            bad_ids = level0_out.loc[level0_out["y_chemprop"].isna(), args.id_col]
            _dump_ids(Path(args.save_dir) / "debug_ids" / "ids_missing_chemprop.csv", bad_ids, "ids_missing_chemprop")
            raise ValueError(
                f"[FATAL] >10% de filas sin y_chemprop tras merge (NA rate={na_rate:.3f}). "
                "Indica desalineación de ids o smiles. Revisa --id-col/--smiles-col y el TEST usado."
            )

    _save_parquet(level0_out, Path(args.save_dir) / "test_level0.parquet")

    # -------------------- blend simple (opcional pesos) ------------------------
    pred_cols = [c for c in ["y_xgb", "y_lgbm", "y_chemprop", "y_tpsa"] if c in level0_out.columns]
    blend_out = level0_out[[args.id_col]].copy()
    if pred_cols:
        if args.weights_json:
            w = _blend_weights_from_json(Path(args.weights_json), pred_cols)
        else:
            w = np.ones(len(pred_cols), dtype=float) / len(pred_cols)
        mat = level0_out[pred_cols].to_numpy(dtype=float)
        y_blend = _blend_with_rowwise_renorm(mat, w)
        blend_out["y_pred_blend"] = y_blend
    _save_parquet(blend_out, Path(args.save_dir) / "test_blend.parquet")

    # -------------------- stack con pesos (opcional) ---------------------------
    if args.weights_json:
        w = _blend_weights_from_json(Path(args.weights_json), pred_cols)
        stack_out = level0_out[[args.id_col]].copy()
        mat = level0_out[pred_cols].to_numpy(dtype=float)
        y_stack = _blend_with_rowwise_renorm(mat, w)
        stack_out["y_pred_stack"] = y_stack
        _save_parquet(stack_out, Path(args.save_dir) / "test_stack.parquet")

    # -------------------- meta-modelo pickle (opcional) ------------------------
    if args.stack_pkl:
        import joblib
        mdl = joblib.load(args.stack_pkl)

        missing = [c for c in pred_cols if c not in level0_out.columns]
        if missing:
            raise ValueError(f"Faltan columnas para meta-modelo: {missing}")

        try:
            y_stack_model = _apply_stack_object(mdl, level0_out, pred_cols)
        except Exception as e:
            raise RuntimeError(f"[FATAL] No se pudo aplicar --stack-pkl: {e}")

        stack_model_out = level0_out[[args.id_col]].copy()
        stack_model_out["y_pred_stack_model"] = y_stack_model
        _save_parquet(stack_model_out, Path(args.save_dir) / "test_stack_model.parquet")

    # -------------------- métricas (si hay y_true en TEST) ---------------------
    metrics = {}
    y_true_col = args.target or "target"
    has_y = y_true_col in test_tab.columns
    if has_y:
        test_tab[y_true_col] = pd.to_numeric(test_tab[y_true_col], errors="coerce")

        merged_eval = test_tab[[args.id_col, y_true_col]].merge(level0_out, on=args.id_col, how="left")
        base_mask = merged_eval[
            [c for c in ["y_xgb", "y_lgbm", "y_tpsa"] if c in merged_eval.columns]
        ].notna().any(axis=1) if set(["y_xgb", "y_lgbm", "y_tpsa"]).intersection(merged_eval.columns) else pd.Series(False, index=merged_eval.index)
        cp_mask = merged_eval["y_chemprop"].notna() if "y_chemprop" in merged_eval.columns else pd.Series(False, index=merged_eval.index)
        inner_mask = base_mask & cp_mask
        union_mask = base_mask | cp_mask

        cohorts = {
            "base_all": int(base_mask.sum()),
            "chemprop": int(cp_mask.sum()),
            "inner": int(inner_mask.sum()),
            "union": int(union_mask.sum()),
        }
        metrics["cohorts"] = cohorts

        def _eval_mask(m, col):
            df = merged_eval.loc[m & merged_eval[col].notna() & merged_eval[y_true_col].notna(), [y_true_col, col]]
            if len(df) == 0:
                return None
            return _metrics(df[y_true_col].values, df[col].values)

        for cohort_name, m in [("base_all", base_mask), ("chemprop", cp_mask),
                               ("inner", inner_mask), ("union", union_mask)]:
            for col in [c for c in ["y_xgb", "y_lgbm", "y_chemprop", "y_tpsa"] if c in merged_eval.columns]:
                res = _eval_mask(m, col)
                if res is not None:
                    metrics.setdefault(cohort_name, {})[col] = res

        df_union = merged_eval[[args.id_col, y_true_col]].copy()

        tb = Path(args.save_dir) / "test_blend.parquet"
        if tb.exists():
            b = pd.read_parquet(tb)
            df = df_union.merge(b, on=args.id_col, how="left")
            mm = df[union_mask & df["y_pred_blend"].notna() & df[y_true_col].notna()]
            if len(mm):
                metrics.setdefault("union", {})["blend"] = _metrics(mm[y_true_col].values, mm["y_pred_blend"].values)

        ts = Path(args.save_dir) / "test_stack.parquet"
        if ts.exists():
            s = pd.read_parquet(ts)
            df = df_union.merge(s, on=args.id_col, how="left")
            mm = df[union_mask & df["y_pred_stack"].notna() & df[y_true_col].notna()]
            if len(mm):
                metrics.setdefault("union", {})["stack"] = _metrics(mm[y_true_col].values, mm["y_pred_stack"].values)

        tsm = Path(args.save_dir) / "test_stack_model.parquet"
        if tsm.exists():
            sm = pd.read_parquet(tsm)
            df = df_union.merge(sm, on=args.id_col, how="left")
            mm = df[union_mask & df["y_pred_stack_model"].notna() & df[y_true_col].notna()]
            if len(mm):
                metrics.setdefault("union", {})["stack_model"] = _metrics(mm[y_true_col].values, mm["y_pred_stack_model"].values)

    if metrics:
        (Path(args.save_dir) / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
        print("[OK] Guardado:", (Path(args.save_dir) / "metrics_test.json").resolve())

    print("[OK] Guardado:")
    print(" -", (Path(args.save_dir) / "test_level0.parquet").resolve())
    print(" -", (Path(args.save_dir) / "test_blend.parquet").resolve())
    if (Path(args.save_dir) / "test_stack.parquet").exists():
        print(" -", (Path(args.save_dir) / "test_stack.parquet").resolve())
    if (Path(args.save_dir) / "test_stack_model.parquet").exists():
        print(" -", (Path(args.save_dir) / "test_stack_model.parquet").resolve())


if __name__ == "__main__":
    main()
