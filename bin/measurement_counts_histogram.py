#!/usr/bin/env python3
# measurement_counts_histogram_single.py
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

def freq_from_counts(counts: pd.Series) -> pd.Series:
    """counts: Serie indexada por molécula con el nº de mediciones; devuelve freq por nº de mediciones."""
    if counts.empty:
        return pd.Series(dtype=int)
    return counts.value_counts().sort_index().astype(int)

def main():
    ap = argparse.ArgumentParser(description="Histograma del nº de mediciones por molécula para un único dataset.")
    ap.add_argument("--input", required=True, help="Parquet de entrada")
    ap.add_argument("--id_col", required=True, help="Columna identificadora de molécula (p.ej. smiles o mol_id)")
    ap.add_argument("--outdir", default="measurements_out", help="Directorio de salida")
    ap.add_argument("--max_bin", type=int, default=None, help="Límite del eje X (máx nº mediciones mostrado)")
    ap.add_argument("--as-perc", dest="as_perc", action="store_true", help="Mostrar porcentaje en lugar de conteo")

    # NUEVO: control fino de ticks
    ap.add_argument("--xminor", type=float, default=1.0,
                    help="Paso de las marcas menores en X (por defecto 1.0). Ej: 0.5, 1, 2…")
    ap.add_argument("--yminor", type=float, default=None,
                    help="Paso de las marcas menores en Y (opcional). Si no se da, se autoajusta.")
    ap.add_argument("--grid-alpha", type=float, default=0.25, help="Transparencia de la rejilla (0-1).")
    ap.add_argument("--grid-ls", default="--", help="Estilo de línea de la rejilla (por defecto --).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.input)
    if args.id_col not in df.columns:
        raise SystemExit(f"[ERROR] '{args.id_col}' no existe en el dataset.")

    # Nº de mediciones por molécula
    counts = df.groupby(args.id_col, dropna=False).size().rename("count_per_mol")
    counts.to_csv(os.path.join(args.outdir, "measurement_counts.csv"))

    # Frecuencia por nº de mediciones
    freq = freq_from_counts(counts)
    ks = freq.index.tolist()

    if args.max_bin is not None:
        ks = [k for k in ks if k <= args.max_bin]
        freq = freq[freq.index.isin(ks)]

    if not ks:
        raise SystemExit("[ERROR] No hay valores para graficar (revisa id_col / datos).")

    # Tabla de salida
    freq_df = freq.reset_index()
    freq_df.columns = ["count_per_mol", "n_molecules"]
    if args.as_perc:
        total = int(freq_df["n_molecules"].sum())
        freq_df["percent"] = (freq_df["n_molecules"] / total * 100.0).round(2)
    freq_df.to_csv(os.path.join(args.outdir, "freq_table.csv"), index=False)

    # Plot
    plt.figure(figsize=(9, 5))
    x_vals = freq_df["count_per_mol"].values
    y_vals = freq_df["percent"].values if args.as_perc else freq_df["n_molecules"].values

    plt.bar(x_vals, y_vals, width=0.9)
    plt.xlabel("Nº de mediciones por molécula")
    plt.ylabel("Porcentaje (%)" if args.as_perc else "Cantidad de moléculas")
    plt.title("Distribución de mediciones por molécula")

    # Límites X (para que la barra 1 quede centrada entre 0.5 y 1.5)
    if args.max_bin:
        plt.xlim(0.5, args.max_bin + 0.5)
    else:
        if len(x_vals) > 0:
            plt.xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)

    ax = plt.gca()

    # Marcas mayores en enteros
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Marcas menores en X (p.ej., cada 1.0 para ver 11, 12, 13… cuando el major fuera 10,20,30)
    if args.xminor and args.xminor > 0:
        ax.xaxis.set_minor_locator(MultipleLocator(args.xminor))

    # Marcas menores en Y si se especifica
    if args.yminor and args.yminor > 0:
        ax.yaxis.set_minor_locator(MultipleLocator(args.yminor))

    # Rejilla en mayores y menores
    ax.grid(which="major", axis="both", linestyle=args.grid-ls if hasattr(args,'grid-ls') else args.grid_ls,
            alpha=args.grid_alpha)
    ax.grid(which="minor", axis="both", linestyle=args.grid-ls if hasattr(args,'grid-ls') else args.grid_ls,
            alpha=args.grid_alpha * 0.8)

    plt.tight_layout()
    out_png = os.path.join(args.outdir, "hist_measurements.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(
        "Listo.\n"
        f"  Figura: {out_png}\n"
        f"  Tabla frec.: {os.path.join(args.outdir,'freq_table.csv')}\n"
        f"  Counts por molécula: {os.path.join(args.outdir,'measurement_counts.csv')}"
    )

if __name__ == "__main__":
    main()
