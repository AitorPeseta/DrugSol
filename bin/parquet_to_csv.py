#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parquet_to_csv.py
-----------------
Convierte archivos Parquet a CSV.
Útil para inspeccionar datos intermedios o exportar resultados finales.
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert Parquet file to CSV format.")
    parser.add_argument("--input", "-i", required=True, help="Ruta al archivo Parquet de entrada")
    parser.add_argument("--output", "-o", required=False, help="Ruta al archivo CSV de salida (Opcional)")
    args = parser.parse_args()

    input_path = Path(args.input)
    
    # Si no se especifica output, usar el mismo nombre pero con extensión .csv
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.csv')

    print(f"[Info] Leyendo {input_path}...")
    
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"[Error] No se pudo leer el archivo Parquet: {e}")
        sys.exit(1)

    print(f"[Info] Filas cargadas: {len(df)}")
    print(f"[Info] Columnas: {list(df.columns)}")

    print(f"[Info] Guardando en {output_path}...")
    try:
        df.to_csv(output_path, index=False)
        print(f"[Éxito] Archivo CSV creado correctamente.")
    except Exception as e:
        print(f"[Error] Fallo al escribir el CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()