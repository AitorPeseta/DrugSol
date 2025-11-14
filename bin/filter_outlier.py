import pandas as pd
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Filtra las filas que NO son outliers (is_outlier == 0) y guarda el resultado."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Archivo de entrada (.parquet o .csv) que contenga la columna 'is_outlier'.")
    parser.add_argument("--save_csv", action="store_true",
                        help="Si se usa, además del Parquet genera también un archivo CSV.")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"No se encontró el archivo: {input_path}")
        sys.exit(1)

    # Leer el archivo según la extensión
    print(f"Leyendo archivo: {input_path}")
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        print("Error: el archivo debe tener extensión .parquet o .csv")
        sys.exit(1)

    if "is_outlier" not in df.columns:
        print("El archivo no contiene la columna 'is_outlier'.")
        sys.exit(1)

    total_rows = len(df)
    outliers = df["is_outlier"].sum()

    # Filtrar solo las filas que NO son outliers
    df_filtered = df[df["is_outlier"] == 0].copy()
    filtered_rows = len(df_filtered)

    # Guardar el resultado en curate.parquet
    output_parquet = "filter_outlier.parquet"
    df_filtered.to_parquet(output_parquet, index=False)
    print(f"Archivo Parquet guardado: {output_parquet}")

    # Guardar CSV si se solicita
    if args.save_csv:
        output_csv = "filter_outlier.csv"
        df_filtered.to_csv(output_csv, index=False)
        print(f"Archivo CSV guardado: {output_csv}")

if __name__ == "__main__":
    main()
