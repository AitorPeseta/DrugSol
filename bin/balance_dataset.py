#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
balance_dataset.py
------------------
Realiza un submuestreo (undersampling) basado en rangos de logS
para equilibrar la distribución de datos.
"""

import argparse
import pandas as pd
import numpy as np
import os

def balance_data(df, target_col='logS', bin_size=1.0, limit_per_bin=100, seed=42):
    print(f"[Balance] Iniciando balanceo. Total inicial: {len(df)}")
    
    # 1. Crear Bins (Cajones)
    # Calculamos min y max para saber dónde empezar
    min_val = np.floor(df[target_col].min())
    max_val = np.ceil(df[target_col].max())
    
    # Creamos los bordes de los bins (ej: -10, -9, -8 ... +2)
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    
    # Asignamos cada molécula a un bin
    # 'include_lowest=True' asegura que el valor mínimo exacto no se quede fuera
    df['bin_id'] = pd.cut(df[target_col], bins=bins, include_lowest=True)
    
    # 2. Función de muestreo por grupo
    def sampler(group):
        n = len(group)
        if n > limit_per_bin:
            # Si hay más del límite, cogemos 'limit_per_bin' aleatorios
            return group.sample(limit_per_bin, random_state=seed)
        else:
            # Si hay menos (ej. insolubles), nos quedamos con todos
            return group

    # 3. Aplicar el balanceo
    # GroupBy por bin y aplicar sampler
    df_balanced = df.groupby('bin_id', observed=True, group_keys=False).apply(sampler)
    
    # 4. Reporte
    print("\n" + "="*40)
    print(f" REPORTE DE BALANCEO (Límite={limit_per_bin}/bin)")
    print("="*40)
    
    # Conteo original vs final por bin
    counts_orig = df['bin_id'].value_counts().sort_index()
    counts_new  = df_balanced['bin_id'].value_counts().sort_index()
    
    for bin_interval in counts_orig.index:
        orig = counts_orig[bin_interval]
        new = counts_new.get(bin_interval, 0)
        if orig > 0:
            print(f"Rango {bin_interval}: {orig} -> {new}")
            
    print("="*40)
    print(f"Total Final: {len(df_balanced)} (Se eliminaron {len(df) - len(df_balanced)} filas)")
    
    # Limpiamos la columna auxiliar
    return df_balanced.drop(columns=['bin_id'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Parquet file")
    ap.add_argument("--output", required=True, help="Output Parquet file")
    ap.add_argument("--limit", type=int, default=100, help="Max samples per log unit bin")
    ap.add_argument("--bin-size", type=float, default=1.0, help="Size of logS bins")
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No existe: {args.input}")
        
    df = pd.read_parquet(args.input)
    
    if 'logS' not in df.columns:
        raise ValueError("El dataset no tiene columna 'logS'")
        
    df_bal = balance_data(df, 'logS', args.bin_size, args.limit, args.seed)
    
    df_bal.to_parquet(args.output, index=False)
    print(f"[Balance] Guardado en: {args.output}")

if __name__ == "__main__":
    main()