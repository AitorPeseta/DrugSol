#!/usr/bin/env python3
# outliers_scatter_mahalanobis_dual.py
# Visualiza train+test en una misma figura. En 1D (p.ej. --only-cols logS)
# permite modos: split (dos bandas), overlay (una banda), jitter (nube).
# En >=2D usa PCA(2) y Mahalanobis multivariante.

import argparse, os
import numpy as np
import pandas as pd

# Backend headless antes de importar pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def mahalanobis_dist(Xz, inv_cov):
    left = Xz @ inv_cov
    d2 = np.einsum('ij,ij->i', left, Xz)
    return np.sqrt(np.clip(d2, 0.0, None))


def fit_inv_cov(Xz):
    cov = np.cov(Xz, rowvar=False)
    return np.linalg.pinv(cov)


def downsample_idx(n, max_points, seed=0):
    idx = np.arange(n)
    if max_points and n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
    return np.sort(idx)


def main():
    ap = argparse.ArgumentParser(
        description="Scatter 2D (PCA) coloreado por Mahalanobis para train+test; "
                    "en 1D (una sola columna) ofrece modos split/overlay/jitter."
    )
    ap.add_argument("--train", required=True, help="Parquet de train")
    ap.add_argument("--test", required=True, help="Parquet de test")
    ap.add_argument("--outlier_col", required=True, help="Columna booleana/0-1 con flags de outliers (en ambos)")
    ap.add_argument("--exclude", nargs="*", default=[], help="Columnas a excluir del cálculo")
    ap.add_argument("--only_cols", dest="only_cols", nargs="*", default=None,
                    help="Usa SOLO estas columnas (ignora el resto)")
    ap.add_argument("--id_col", default=None, help="Columna identificadora (opcional) para el CSV")
    ap.add_argument("--basis", choices=["train","combined"], default="train",
                    help="Base para scaler/cov/PCA")
    ap.add_argument("--max_points", type=int, default=50000, help="Máx. puntos totales a dibujar")
    ap.add_argument("--alpha", type=float, default=0.6, help="Transparencia de puntos")
    ap.add_argument("--size", type=float, default=12, help="Tamaño de puntos")
    ap.add_argument("--outdir", default="outlier_viz", help="Directorio de salida")

    # Opciones 1D
    ap.add_argument("--y-mode", choices=["split","overlay","jitter"], default="split",
                    help="En 1D: split (dos bandas), overlay (una), jitter (nube)")
    ap.add_argument("--jitter-amp", type=float, default=0.15,
                    help="Amplitud del jitter vertical si y-mode=jitter")
    ap.add_argument("--thresh", type=float, default=None,
                    help="Traza líneas en ±thresh de z-score en 1D (p.ej. 3.0)")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tr = pd.read_parquet(args.train)
    te = pd.read_parquet(args.test)

    # Validaciones
    if args.outlier_col not in tr.columns or args.outlier_col not in te.columns:
        raise SystemExit(f"[ERROR] La columna '{args.outlier_col}' debe existir en train y test.")
    if args.id_col and (args.id_col not in tr.columns or args.id_col not in te.columns):
        raise SystemExit(f"[ERROR] id_col '{args.id_col}' no existe en ambos datasets.")

    # Selección de features numéricas comunes
    common = sorted(set(tr.columns) & set(te.columns))
    if args.only_cols:
        feats = [c for c in args.only_cols if c in common]
        missing = sorted(set(args.only_cols) - set(feats))
        if missing:
            print(f"[AVISO] No existen en ambos y se omiten: {missing}")
    else:
        feats = [c for c in common if c not in set(args.exclude + [args.outlier_col])]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(tr[c]) and pd.api.types.is_numeric_dtype(te[c])]

    if not feats:
        raise SystemExit("[ERROR] No hay columnas numéricas seleccionadas y comunes.")

    # Matrices y máscaras de NaN
    Xtr_raw = tr[feats].to_numpy()
    Xte_raw = te[feats].to_numpy()
    mtr = np.isfinite(Xtr_raw).all(axis=1)
    mte = np.isfinite(Xte_raw).all(axis=1)
    Xtr_raw, Xte_raw = Xtr_raw[mtr], Xte_raw[mte]
    out_tr = tr.loc[mtr, args.outlier_col].astype(int).to_numpy()
    out_te = te.loc[mte, args.outlier_col].astype(int).to_numpy()
    idtr = tr.loc[mtr, args.id_col].to_numpy() if args.id_col else None
    idte = te.loc[mte, args.id_col].to_numpy() if args.id_col else None

    # Estandarización
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(Xtr_raw if args.basis == "train" else np.vstack([Xtr_raw, Xte_raw]))
    Ztr, Zte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    # --- Caso 1D: MD = |z|; modos split/overlay/jitter ---
    if Ztr.shape[1] == 1:
        ztr = Ztr[:, 0]
        zte = Zte[:, 0]
        md_tr = np.abs(ztr)
        md_te = np.abs(zte)

        # y-coords según modo
        ntr, nte = ztr.size, zte.size
        if args.y_mode == "split":
            y_tr = np.zeros(ntr)
            y_te = np.ones(nte)
        elif args.y_mode == "overlay":
            y_tr = np.zeros(ntr)
            y_te = np.zeros(nte)
        else:  # jitter
            rng = np.random.default_rng(0)
            y_tr = rng.normal(0.0, args.jitter_amp, ntr)
            y_te = rng.normal(0.0, args.jitter_amp, nte)

        # Muestreo (reparte presupuesto entre splits)
        half = args.max_points // 2 if args.max_points else None
        idx_tr = downsample_idx(ntr, half or ntr)
        idx_te = downsample_idx(nte, (args.max_points - idx_tr.size) if args.max_points else nte)

        ztr_p, ytr_p, mdtr_p, otr_p = ztr[idx_tr], y_tr[idx_tr], md_tr[idx_tr], out_tr[idx_tr].astype(bool)
        zte_p, yte_p, mdte_p, ote_p = zte[idx_te], y_te[idx_te], md_te[idx_te], out_te[idx_te].astype(bool)

        # Colores compartidos
        cmin = float(min(mdtr_p.min() if mdtr_p.size else 0, mdte_p.min() if mdte_p.size else 0))
        cmax = float(max(mdtr_p.max() if mdtr_p.size else 1, mdte_p.max() if mdte_p.size else 1))
        if cmin == cmax:
            cmax = cmin + 1e-9

        plt.figure(figsize=(9, 4.2))
        # Base
        sc1 = plt.scatter(ztr_p, ytr_p, c=mdtr_p, vmin=cmin, vmax=cmax,
                          s=args.size, alpha=args.alpha, marker='o', label="train")
        sc2 = plt.scatter(zte_p, yte_p, c=mdte_p, vmin=cmin, vmax=cmax,
                          s=args.size, alpha=args.alpha, marker='^', label="test")

        # Aros de outlier con el MISMO marcador de cada split
        if otr_p.any():
            plt.scatter(ztr_p[otr_p], ytr_p[otr_p],
                        facecolors='none', edgecolors='k',
                        marker='o', s=args.size*1.8, linewidths=0.9, zorder=3)
        if ote_p.any():
            plt.scatter(zte_p[ote_p], yte_p[ote_p],
                        facecolors='none', edgecolors='k',
                        marker='^', s=args.size*1.8, linewidths=0.9, zorder=3)

        # Líneas de umbral ±thresh (opcional)
        if args.thresh is not None:
            plt.axvline(+args.thresh, ls="--", lw=1, color="gray")
            plt.axvline(-args.thresh, ls="--", lw=1, color="gray")

        cb = plt.colorbar(sc1 if ztr_p.size else sc2)
        cb.set_label("|z-score| (Mahalanobis 1D)")

        if args.y_mode == "split":
            plt.yticks([0, 1], ["train", "test"])
        else:
            plt.yticks([])

        title_mode = {"split": "dos bandas", "overlay": "unido", "jitter": "jitter"}[args.y_mode]
        plt.xlabel(f"z-score de {feats[0]}  (base={args.basis})")
        plt.title(f"Outliers por {feats[0]} (Mahalanobis 1D = |z|) • {title_mode}")
        plt.legend(loc="best")
        plt.tight_layout()

        out_png = os.path.join(args.outdir, "scatter_mahalanobis.png")
        plt.savefig(out_png, dpi=150)
        plt.close()

        # CSV con distancias para TODAS las filas válidas (sin muestreo)
        rows = []
        for i in range(md_tr.size):
            r = {"split": "train", "mahalanobis": float(md_tr[i]), "is_outlier": int(out_tr[i])}
            if args.id_col:
                r[args.id_col] = idtr[i]
            rows.append(r)
        for i in range(md_te.size):
            r = {"split": "test", "mahalanobis": float(md_te[i]), "is_outlier": int(out_te[i])}
            if args.id_col:
                r[args.id_col] = idte[i]
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "mahalanobis.csv"), index=False)

        print(f"Listo. Gráfico: {out_png}  •  Distancias: {os.path.join(args.outdir, 'mahalanobis.csv')}")
        return

    # --- Caso multivariante (>=2D): Mahalanobis general + PCA 2D ---
    inv_cov = fit_inv_cov(Ztr if args.basis == "train" else np.vstack([Ztr, Zte]))
    md_tr = mahalanobis_dist(Ztr, inv_cov)
    md_te = mahalanobis_dist(Zte, inv_cov)

    pca = PCA(n_components=2, random_state=0)
    pca.fit(Ztr if args.basis == "train" else np.vstack([Ztr, Zte]))
    XYtr, XYte = pca.transform(Ztr), pca.transform(Zte)

    # Muestreo
    half = args.max_points // 2 if args.max_points else None
    idx_tr = downsample_idx(XYtr.shape[0], half or XYtr.shape[0])
    idx_te = downsample_idx(XYte.shape[0], (args.max_points - idx_tr.size) if args.max_points else XYte.shape[0])

    xtr, ytr2, ctr, oftr = XYtr[idx_tr, 0], XYtr[idx_tr, 1], md_tr[idx_tr], out_tr[idx_tr].astype(bool)
    xte, yte2, cte, ofte = XYte[idx_te, 0], XYte[idx_te, 1], md_te[idx_te], out_te[idx_te].astype(bool)

    cmin = float(min(ctr.min() if ctr.size else 0, cte.min() if cte.size else 0))
    cmax = float(max(ctr.max() if ctr.size else 1, cte.max() if cte.size else 1))
    if cmin == cmax:
        cmax = cmin + 1e-9

    plt.figure(figsize=(8, 7))
    if xtr.size:
        sc_tr = plt.scatter(xtr, ytr2, c=ctr, vmin=cmin, vmax=cmax,
                            s=args.size, alpha=args.alpha, marker='o', label="train")
        if oftr.any():
            plt.scatter(xtr[oftr], ytr2[oftr],
                        facecolors='none', edgecolors='k',
                        marker='o', s=args.size*1.8, linewidths=0.9, zorder=3)
    if xte.size:
        sc_te = plt.scatter(xte, yte2, c=cte, vmin=cmin, vmax=cmax,
                            s=args.size, alpha=args.alpha, marker='^', label="test")
        if ofte.any():
            plt.scatter(xte[ofte], yte2[ofte],
                        facecolors='none', edgecolors='k',
                        marker='^', s=args.size*1.8, linewidths=0.9, zorder=3)

    plt.colorbar(sc_tr if xtr.size else sc_te).set_label("Distancia de Mahalanobis")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    plt.title(f"PCA coloreado por Mahalanobis • base={args.basis}")
    plt.legend(loc="best"); plt.tight_layout()
    out_png = os.path.join(args.outdir, "scatter_mahalanobis.png")
    plt.savefig(out_png, dpi=150); plt.close()

    # CSV con distancias (todas las filas válidas)
    rows = []
    for i in range(Ztr.shape[0]):
        r = {"split": "train", "mahalanobis": float(md_tr[i]), "is_outlier": int(out_tr[i])}
        if args.id_col:
            r[args.id_col] = idtr[i]
        rows.append(r)
    for i in range(Zte.shape[0]):
        r = {"split": "test", "mahalanobis": float(md_te[i]), "is_outlier": int(out_te[i])}
        if args.id_col:
            r[args.id_col] = idte[i]
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "mahalanobis.csv"), index=False)

    print(f"Listo. Gráfico: {out_png}  •  Distancias: {os.path.join(args.outdir, 'mahalanobis.csv')}")


if __name__ == "__main__":
    main()
