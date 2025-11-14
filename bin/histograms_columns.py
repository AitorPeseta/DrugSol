#!/usr/bin/env python3
import argparse, os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def load_cols_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]

def pick_columns(df, cols_cli, cols_file, cols_regex, exclude):
    wanted = set()
    if cols_cli: wanted.update(cols_cli)
    if cols_file: wanted.update(load_cols_file(cols_file))
    if cols_regex:
        pat = re.compile(cols_regex)
        wanted.update([c for c in df.columns if pat.search(c)])

    # --- alias rápido: si pides 'mordred__MW', incluye también columnas 'MW' afines ---
    if "mordred__MW" in wanted:
        mw_like = {
            c for c in df.columns
            if c == "MW" or c.lower() == "mw" or c.endswith("_MW") or c.endswith("__MW")
        }
        wanted.update(mw_like)

    if not wanted:
        print("[ERROR] No se especificaron columnas con --cols / --cols-file / --cols-regex.", file=sys.stderr)
        sys.exit(2)

    existing = set(df.columns)
    missing = sorted(list(wanted - existing))
    selected = sorted(list(wanted & existing))
    if missing:
        print(f"[AVISO] Columnas no encontradas y que se ignoran: {missing}", file=sys.stderr)
    selected = [c for c in selected if c not in set(exclude or [])]
    return selected


def safe_log1p(arr):
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= -1]
    return np.log1p(arr)

def apply_rounding(arr, round_step=None, round_decimals=None):
    if arr.size == 0:
        return arr
    if round_step is not None and round_step > 0:
        return np.round(arr / round_step) * round_step
    if round_decimals is not None and round_decimals >= 0:
        return np.round(arr, int(round_decimals))
    return arr

def convert_temp_kelvin_to_celsius_if_needed(colname, arr, auto=True, extra_cols=None):
    """
    Si auto=True, convierte si el nombre es 'temperature_K' o coincide con patrones comunes de Kelvin.
    También convierte si colname está en extra_cols.
    """
    if arr.size == 0:
        return arr, colname

    extra_cols = set(extra_cols or [])
    lower = colname.lower()

    should_convert = False
    if auto:
        # disparadores comunes
        if colname == "temperature_K":
            should_convert = True
        elif lower.endswith("_k") and "temperature" in lower:
            should_convert = True
        elif lower in {"temp_k", "temperaturek"}:
            should_convert = True

    if colname in extra_cols:
        should_convert = True

    if should_convert:
        # K → °C
        arr = arr - 273.15
        # renombrar etiqueta del eje
        # intenta mantener el prefijo del nombre
        new_label = re.sub(r'(_?K)$', '_°C', colname, flags=re.IGNORECASE)
        if new_label == colname:
            # si no cambió, al menos añade sufijo
            new_label = f"{colname}_°C"
        return arr, new_label
    return arr, colname

def main():
    ap = argparse.ArgumentParser(
        description="Histogramas para columnas seleccionadas combinando train+test u overlay, con ticks menores, rejilla y conversión automática de temperature_K a °C."
    )
    ap.add_argument("--train", required=True, help="Parquet de train")
    ap.add_argument("--test", required=True, help="Parquet de test")
    ap.add_argument("--outdir", default="hist_out", help="Directorio de salida")
    ap.add_argument("--bins", type=int, default=50, help="Número de bins")
    ap.add_argument("--density", action="store_true", help="Normaliza a densidad (área=1)")
    ap.add_argument("--log1p", action="store_true", help="Aplica log1p")
    ap.add_argument("--overlay", action="store_true", help="Superpone train y test con leyenda")
    # redondeo
    ap.add_argument("--round-decimals", type=int, default=None, help="Redondea a N decimales (ej. 2)")
    ap.add_argument("--round-step", type=float, default=None, help="Redondea al múltiplo más cercano de STEP (ej. 0.1)")
    ap.add_argument("--round-before-log", action="store_true", help="Aplica redondeo antes de log1p (por defecto se redondea después)")
    # selección de columnas
    ap.add_argument("--cols", nargs="*", help="Lista de columnas a graficar")
    ap.add_argument("--cols-file", help="Archivo con nombres de columnas (una por línea)")
    ap.add_argument("--cols-regex", help="Regex para seleccionar columnas por nombre")
    ap.add_argument("--exclude", nargs="*", default=[], help="Columnas a excluir")
    # ejes, rejilla, ticks menores
    ap.add_argument("--no-grid", dest="grid", action="store_false", help="Desactiva la rejilla")
    ap.set_defaults(grid=True)
    ap.add_argument("--no-minor-ticks", dest="minor_ticks", action="store_false", help="Desactiva los ticks menores")
    ap.set_defaults(minor_ticks=True)
    ap.add_argument("--minor-divs", type=int, default=5, help="Subdivisiones menores por intervalo mayor (por defecto 5)")
    # conversión temperature K→°C
    ap.add_argument("--no-auto-temp-k-to-c", dest="auto_temp_k_to_c", action="store_false",
                    help="No convertir automáticamente columnas de temperatura en Kelvin a °C")
    ap.set_defaults(auto_temp_k_to_c=True)
    ap.add_argument("--temp-k-cols", nargs="*", default=[], help="Nombres adicionales de columnas a convertir de K a °C")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_tr = pd.read_parquet(args.train)
    df_te = pd.read_parquet(args.test)

    # columnas comunes (para validar selección)
    common_cols = list(sorted(set(df_tr.columns) & set(df_te.columns)))
    df_common = pd.DataFrame(columns=common_cols)
    selected = pick_columns(df_common, args.cols, args.cols_file, args.cols_regex, args.exclude)

    # filtra a numéricas presentes en ambos
    num_cols, nonnum_warn = [], []
    for c in selected:
        if c in df_tr.columns and c in df_te.columns \
           and pd.api.types.is_numeric_dtype(df_tr[c]) and pd.api.types.is_numeric_dtype(df_te[c]):
            num_cols.append(c)
        else:
            nonnum_warn.append(c)
    if nonnum_warn:
        print(f"[AVISO] Columnas no numéricas o ausentes en alguno de los datasets que se omiten: {nonnum_warn}", file=sys.stderr)
    if not num_cols:
        print("[ERROR] Ninguna columna seleccionada es numérica y común a train y test.", file=sys.stderr)
        sys.exit(2)

    rows = []
    for c in num_cols:
        s_tr = df_tr[c].dropna().values
        s_te = df_te[c].dropna().values

        # Conversión de temperatura si procede
        s_tr, x_label_tr = convert_temp_kelvin_to_celsius_if_needed(
            c, s_tr, auto=args.auto_temp_k_to_c, extra_cols=args.temp_k_cols
        )
        s_te, x_label_te = convert_temp_kelvin_to_celsius_if_needed(
            c, s_te, auto=args.auto_temp_k_to_c, extra_cols=args.temp_k_cols
        )
        # etiqueta final del eje X
        x_label = x_label_tr if x_label_tr == x_label_te else c  # si difieren, conserva nombre original

        # Redondeo/transformación en el orden solicitado
        if args.round_before_log:
            s_tr = apply_rounding(s_tr, args.round_step, args.round_decimals)
            s_te = apply_rounding(s_te, args.round_step, args.round_decimals)

        if args.log1p:
            s_tr = safe_log1p(s_tr)
            s_te = safe_log1p(s_te)

        if not args.round_before_log:
            s_tr = apply_rounding(s_tr, args.round_step, args.round_decimals)
            s_te = apply_rounding(s_te, args.round_step, args.round_decimals)

        # descarta NaN/Inf por si acaso tras transformaciones
        s_tr = s_tr[np.isfinite(s_tr)]
        s_te = s_te[np.isfinite(s_te)]

        # stats helper
        def stats(arr):
            if arr.size == 0:
                return dict(count=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
            return dict(
                count=int(arr.size),
                mean=float(np.nanmean(arr)),
                std=float(np.nanstd(arr, ddof=1)) if arr.size > 1 else np.nan,
                min=float(np.nanmin(arr)),
                max=float(np.nanmax(arr)),
            )

        st_tr, st_te = stats(s_tr), stats(s_te)

        # Graficado
        fig, ax = plt.subplots(figsize=(8, 5))
        if args.overlay:
            # rango común para comparar
            if st_tr["count"] and st_te["count"]:
                xmin = min(st_tr["min"], st_te["min"])
                xmax = max(st_tr["max"], st_te["max"])
                if xmin == xmax: xmax = xmin + 1e-9
            elif st_tr["count"]:
                xmin, xmax = st_tr["min"], st_tr["max"]
                if xmin == xmax: xmax = xmin + 1e-9
            else:
                xmin, xmax = st_te["min"], st_te["max"]
                if xmin == xmax: xmax = xmin + 1e-9
            if st_tr["count"]:
                ax.hist(s_tr, bins=args.bins, range=(xmin, xmax), alpha=0.5, density=args.density, label="train")
            if st_te["count"]:
                ax.hist(s_te, bins=args.bins, range=(xmin, xmax), alpha=0.5, density=args.density, label="test")
            ax.legend(loc="best")
        else:
            data_all = np.concatenate([s_tr, s_te])
            ax.hist(data_all, bins=args.bins, density=args.density)

        # Ticks menores y rejilla
        if args.minor_ticks:
            try:
                ax.xaxis.set_minor_locator(AutoMinorLocator(args.minor_divs))
                ax.yaxis.set_minor_locator(AutoMinorLocator(args.minor_divs))
            except Exception:
                # AutoMinorLocator no disponible en algunas versiones antiguas
                pass
        if args.grid:
            ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.6)
            ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

        # títulos/labels
        ax.set_title(f"Histograma: {c}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Densidad" if args.density else "Frecuencia")

        out_png = os.path.join(args.outdir, f"{c}.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        rows.append({
            "column": c,
            "x_label": x_label,
            "mode": "overlay" if args.overlay else "combined",
            "image": out_png,
            "train_count": st_tr["count"], "train_mean": st_tr["mean"], "train_std": st_tr["std"],
            "train_min": st_tr["min"], "train_max": st_tr["max"],
            "test_count": st_te["count"], "test_mean": st_te["mean"], "test_std": st_te["std"],
            "test_min": st_te["min"], "test_max": st_te["max"],
            "round_step": args.round_step, "round_decimals": args.round_decimals,
            "rounded_before_log": bool(args.round_before_log),
            "density": bool(args.density),
            "bins": int(args.bins),
            "minor_ticks": bool(args.minor_ticks),
            "minor_divs": int(args.minor_divs),
            "grid": bool(args.grid),
            "auto_temp_k_to_c": bool(args.auto_temp_k_to_c)
        })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "hist_index.csv"), index=False)
        print(f"Listo. {len(rows)} histogramas en {args.outdir}")
    else:
        print("[AVISO] No se generó ningún histograma.", file=sys.stderr)

if __name__ == "__main__":
    main()
