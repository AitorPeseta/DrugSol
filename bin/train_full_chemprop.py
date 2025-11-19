#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, subprocess, tempfile, shutil
from pathlib import Path
import pandas as pd
import numpy as np

def _accel_flags(use_gpu: bool):
    try:
        import torch
        if use_gpu and torch.cuda.is_available():
            return ["--accelerator", "gpu", "--devices", "1"]
    except Exception:
        pass
    return ["--accelerator", "cpu", "--devices", "1"]

def run(cmd):
    print("[INFO] Lanzando:", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.stdout: print(p.stdout)
    if p.stderr: print(p.stderr)
    if p.returncode != 0:
        raise SystemExit("chemprop command failed")
    return p

def load_best_params(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

def parse_args():
    ap = argparse.ArgumentParser("Fase 3 · full train Chemprop")
    ap.add_argument("--train", required=True, help="parquet/csv con smiles + target (+ opcional descriptores)")
    ap.add_argument("--smiles-col", default="smiles_neutral")
    ap.add_argument("--target", default="logS")
    ap.add_argument("--save-dir", default="models/chemprop")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)  # early stopping por si val es pequeña
    ap.add_argument("--val-fraction", type=float, default=0.05, help="fracción para validación (mínimo 1 fila)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--best-params", default="oof_gnn/chemprop_best_params.json")

    # overrides rápidos
    ap.add_argument("--hidden-size", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--ffn-num-layers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)

    # Carga datos
    df = pd.read_parquet(args.train) if str(args.train).endswith(".parquet") else pd.read_csv(args.train)
    assert args.smiles_col in df.columns, f"Falta {args.smiles_col}"
    assert args.target in df.columns, f"Falta target {args.target}"

    # Subconjunto limpio y barajado reproducible
    df = df.dropna(subset=[args.smiles_col, args.target]).copy()
    df = df[df[args.smiles_col].astype(str).str.len() > 0]
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Filtrar descriptores explícitos a los que realmente existan y sean numéricos
    present_desc = [c for c in ["temp_C","n_ionizable","n_acid","n_base",
                                "TPSA","logP","HBD","HBA","FractionCSP3","MW"]
                    if c in df.columns]

    n = len(df)
    if n < 2:
        raise SystemExit("Dataset demasiado pequeño para reservar validación (necesita al menos 2 filas).")

    # Hiperparámetros
    hp = load_best_params(args.best_params)
    if args.hidden_size is not None:    hp["hidden_size"]    = args.hidden_size
    if args.depth is not None:          hp["depth"]          = args.depth
    if args.dropout is not None:        hp["dropout"]        = args.dropout
    if args.ffn_num_layers is not None: hp["ffn_num_layers"] = args.ffn_num_layers
    if args.batch_size is not None:     hp["batch_size"]     = args.batch_size

    epochs         = int(hp.get("epochs", args.epochs))
    hidden_size    = int(hp.get("hidden_size", 1600))
    depth          = int(hp.get("depth", 3))
    dropout        = float(hp.get("dropout", 0.10))
    ffn_num_layers = int(hp.get("ffn_num_layers", 1))
    batch_size     = int(hp.get("batch_size", 32))

    # Split explícito con al menos 1 en validación
    vf = max(args.val_fraction, 1.0 / n)
    n_val   = max(1, int(round(n * vf)))
    n_train = n - n_val
    if n_train <= 0: n_train, n_val = n - 1, 1

    with tempfile.TemporaryDirectory(prefix="chemprop_full_") as td:
        td = Path(td)
        comb_csv = td / "combined.csv"

        # Renombrar smiles -> 'smiles' y dejar target tal cual
        rename_map = {args.smiles_col: "smiles"}
        df_comb = df.rename(columns=rename_map).copy()
        df_comb.to_csv(comb_csv, index=False)

        splits = [{
            "train": list(range(0, n_train)),
            "val":   list(range(n_train, n)),
            "test":  []
        }]
        splits_file = td / "splits.json"
        splits_file.write_text(json.dumps(splits))

        run_dir = outdir / "run_full"
        if run_dir.exists(): shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Comando Chemprop (sin --descriptors-path; solo columnas ya en CSV)
        cmd = [
            "chemprop","train",
            "-i", str(comb_csv),
            "--splits-file", str(splits_file),
            "--smiles-columns","smiles",
            "--target-columns", args.target,
            "-o", str(run_dir),
            "--epochs", str(epochs),
            "--patience", str(args.patience),
            "--metric","rmse",
            "--tracking-metric","val_loss",
            "--batch-size", str(batch_size),
            "--message-hidden-dim", str(hidden_size),
            "--depth", str(depth),
            "--dropout", str(dropout),
            "--ffn-num-layers", str(ffn_num_layers),
            *(_accel_flags(args.gpu)),
        ]
        if present_desc:
            cmd += ["--descriptors-columns", *present_desc]

        run(cmd)

        # Buscar best.pt
        best_ckpt = next((p for p in run_dir.rglob("best.pt")), None)
        if best_ckpt is None:
            pts = list(run_dir.rglob("*.pt"))
            if not pts: raise FileNotFoundError("No se encontró ningún checkpoint .pt del modelo entrenado.")
            best_ckpt = pts[0]

        final_pt = outdir / "best.pt"
        shutil.copy2(best_ckpt, final_pt)
        print(f"[OK] Modelo final guardado en: {final_pt.resolve()}")

        # ===== Evaluación interna en la partición de validación =====
        # Construimos el CSV de entrada con las mismas columnas de descriptores
        val_df = df_comb.iloc[n_train:].copy()
        val_in_csv  = run_dir / "val_in.csv"
        use_cols = ["smiles"] + present_desc
        val_df[use_cols].to_csv(val_in_csv, index=False)

        val_out_csv = run_dir / "val_pred.csv"
        cmd_pred = [
            "chemprop","predict",
            "-i", str(val_in_csv),
            "-o", str(val_out_csv),
            "--model-paths", str(final_pt),
            "--drop-extra-columns",
            *(_accel_flags(args.gpu)),
        ]
        if present_desc:
            cmd_pred += ["--descriptors-columns", *present_desc]
        run(cmd_pred)

        pred = pd.read_csv(val_out_csv)
        # detectar columna de pred
        lower = {c.lower(): c for c in pred.columns}
        if "prediction" in lower:
            pred_col = lower["prediction"]
        elif args.target.lower() in lower:
            pred_col = lower[args.target.lower()]
        elif "value" in lower:
            pred_col = lower["value"]
        else:
            cand = [c for c in pred.columns if c.lower() not in ("smiles", *[d.lower() for d in present_desc])]
            pred_col = cand[-1] if cand else pred.columns[-1]

        y_true = df.iloc[n_train:][args.target].to_numpy()
        y_hat  = pred[pred_col].to_numpy()
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = mean_squared_error(y_true, y_hat, squared=False)
        r2   = r2_score(y_true, y_hat)
        print(f"[VAL] RMSE={rmse:.4f}  R2={r2:.4f}  (n={len(y_true)})")

    # Manifiesto
    manifest = {
        "epochs": epochs,
        "hidden_size": hidden_size,
        "depth": depth,
        "dropout": dropout,
        "ffn_num_layers": ffn_num_layers,
        "batch_size": batch_size,
        "target": args.target,
        "smiles_col": args.smiles_col,
        "val_fraction_effective": float(max(args.val_fraction, 1.0/len(df))),
        "used_gpu": bool(args.gpu),
        "best_params_source": str(args.best_params),
        "descriptors_used": present_desc,
        "seed": int(args.seed),
    }
    (outdir / "chemprop_manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
