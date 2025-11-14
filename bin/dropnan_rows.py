#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import re

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".csv":
        # Mantén el parser estándar; los vacíos los trataremos nosotros
        return pd.read_csv(path)
    else:
        print("Extensión no soportada. Usa .parquet o .csv")
        sys.exit(1)

def write_any(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        print("Extensión no soportada para salida. Usa .parquet o .csv")
        sys.exit(1)

def coerce_empty_strings_to_nan(df: pd.DataFrame, only_object_cols: bool = True) -> pd.DataFrame:
    """
    Convierte cadenas vacías o de solo espacios en NaN.
    Por defecto, sólo aplica a columnas de tipo 'object'/'string' para no tocar números.
    """
    df2 = df.copy()
    if only_object_cols:
        cols = df2.select_dtypes(include=["object", "string"]).columns.tolist()
    else:
        cols = df2.columns.tolist()

    if not cols:
        return df2

    # strip y marcar vacíos como NaN
    df2[cols] = df2[cols].applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df2[cols] = df2[cols].replace(to_replace=r"^\s*$", value=np.nan, regex=True)
    return df2

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Limpieza de NaNs sin perder columnas:\n"
            " - mode=train: elimina filas SOLO si se especifica --subset (p. ej., el target).\n"
            " - mode=test : no elimina filas por NaN del subset, pero sí quita filas con strings vacíos.\n"
            "Además: convierte strings vacíos (ej. ',,') en NaN y elimina esas filas siempre."
        )
    )
    ap.add_argument("--input", "-i", required=True, help="Archivo de entrada (.parquet o .csv)")
    ap.add_argument("--output", "-o", required=True, help="Archivo de salida (.parquet o .csv)")
    ap.add_argument("--subset", nargs="*", default=None,
                    help="Columnas a considerar para dropna de filas (solo en mode=train). Si no se indica, no se eliminan filas por subset.")
    ap.add_argument("--save_csv", action="store_true",
                    help="Además del --output, guarda también un .csv si --output es parquet.")
    ap.add_argument("--mode", choices=["train", "test"], default="train",
                    help="train: dropna por filas SOLO en --subset; test: no elimina filas por subset.")
    ap.add_argument("--keep_all_columns", action="store_true",
                    help="(Por compatibilidad) No se eliminan columnas nunca (por defecto ya es así).")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"No se encontró: {args.input}")
        sys.exit(1)

    df = read_any(args.input)
    n0 = len(df)

    # Normaliza inf a NaN (no cambia número de columnas)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 1) Convertir strings vacíos/espacios en NaN SOLO en columnas de texto
    df1 = coerce_empty_strings_to_nan(df, only_object_cols=True)

    # 2) Eliminar SIEMPRE filas con strings vacíos (ya convertidos a NaN) en columnas de texto
    object_cols = df1.select_dtypes(include=["object", "string"]).columns.tolist()
    if object_cols:
        n_before_empty = len(df1)
        df1 = df1.dropna(axis=0, how="any", subset=object_cols)
        dropped_empty = n_before_empty - len(df1)
    else:
        dropped_empty = 0

    # 3) Comportamiento por modo:
    if args.mode == "test":
        # En TEST no eliminamos más filas por subset (pero ya quitamos las vacías de texto arriba)
        df2 = df1
        action = "TEST (quitados vacíos en columnas de texto; sin dropna por subset)"
    else:
        # TRAIN: eliminar filas por NaN en subset (si se indicó)
        if args.subset:
            missing = [c for c in args.subset if c not in df1.columns]
            if missing:
                print(f"Columnas no encontradas en --subset: {missing}")
                sys.exit(1)
            n_before_subset = len(df1)
            df2 = df1.dropna(axis=0, how="any", subset=args.subset)
            dropped_subset = n_before_subset - len(df2)
            action = f"TRAIN (quitados vacíos en texto: {dropped_empty} | dropna por subset={args.subset}: {dropped_subset})"
        else:
            df2 = df1
            action = f"TRAIN (quitados vacíos en texto: {dropped_empty} | sin subset → no se eliminan más filas)"

    n1 = len(df2)
    dropped_total = n0 - n1
    print(f"{action} | Filas iniciales: {n0} | Filas finales: {n1} | Filas eliminadas totales: {dropped_total}")

    if n1 == 0:
        print("Tras la limpieza no quedan filas. No se escribe salida para evitar un archivo vacío.")
        sys.exit(2)

    write_any(df2, args.output)

    if args.save_csv and args.output.lower().endswith(".parquet"):
        out_csv = os.path.splitext(args.output)[0] + ".csv"
        df2.to_csv(out_csv, index=False)
        print(f"Guardado CSV: {out_csv}")

if __name__ == "__main__":
    main()
