#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
balance_dataset_smart_2d.py
---------------------------
Balanceo inteligente:
1. Identifica la temperatura del grupo.
2. Si es Temperatura Ambiente (20-30°C) -> APLICA LÍMITE (Undersampling).
3. Si es Temperatura Fisiológica o Alta -> GUARDA TODO (100%).
4. Permite definir el tamaño del bin de logS (ej: 0.2).
"""

import argparse
import pandas as pd
import numpy as np
import os

def balance_data_smart_2d(df, log_col='logS', temp_col='temp_C', 
                          limit_ambient=50, 
                          bin_size=0.2,
                          seed=42):
    
    print(f"[Balance Smart] Iniciando.")
    print(f"                - Bin Size (logS): {bin_size}")
    print(f"                - Datos 25°C (Ambient): Recortar a {limit_ambient} por bin.")
    print(f"                - Datos >30°C o <20°C:  GUARDAR TODOS.")

    # 1. Bins de Solubilidad (Ahora usa tu bin_size variable)
    min_log = np.floor(df[log_col].min())
    max_log = np.ceil(df[log_col].max())
    # Importante: Añadir bin_size al final para asegurar que se cubra el máximo
    bins_log = np.arange(min_log, max_log + bin_size, bin_size)
    
    # 2. Bins de Temperatura
    bins_temp = [24, 25, 35, 40, 50]
    labels_temp = ['Ambient', 'Normal', 'Physio', 'Hot_Physio']
    
    # Asignamos los bins
    df['bin_log'] = pd.cut(df[log_col], bins=bins_log, include_lowest=True)
    df['bin_temp'] = pd.cut(df[temp_col], bins=bins_temp, labels=labels_temp, include_lowest=True)
    
    # 3. Función de muestreo condicional
    def sampler(group):
        temp_label = group.name[1] 
        
        # Si es Ambiente, aplicamos el límite
        if temp_label == 'Ambient':
            n = len(group)
            if n > limit_ambient:
                return group.sample(limit_ambient, random_state=seed)
            else:
                return group
        # Si es cualquier otra cosa (Fisio, Calor, Frío), guardamos TODO
        else:
            return group

    # 4. Ejecutar GroupBy
    # Al ser bins más pequeños (0.2), habrá muchos más grupos, lo cual es bueno
    # porque el recorte será más distribuido y menos "bloque".
    df_balanced = df.groupby(['bin_log', 'bin_temp'], observed=True, group_keys=False).apply(sampler)
    
    # 5. Reporte
    print("\n" + "="*60)
    print(f" REPORTE FINAL (Bin Size: {bin_size})")
    print("="*60)
    
    orig_counts = df['bin_temp'].value_counts()
    new_counts = df_balanced['bin_temp'].value_counts()
    
    for label in labels_temp:
        if label in orig_counts:
            o = orig_counts[label]
            n = new_counts.get(label, 0)
            status = "Recortado" if label == 'Ambient' else "Intacto"
            print(f"  - {label:12s}: {o:6d} -> {n:6d}  [{status}]")
            
    print("-" * 60)
    print(f"Total: {len(df)} -> {len(df_balanced)}")
    
    return df_balanced.drop(columns=['bin_log', 'bin_temp'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    # Argumentos numéricos
    ap.add_argument("--limit", type=int, default=50, help="Límite máx para muestras Ambient")
    ap.add_argument("--bin-size", type=float, default=0.2, help="Tamaño del bin de logS (Default: 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No existe: {args.input}")
    
    df = pd.read_parquet(args.input)
    
    if df['temp_C'].isnull().any():
        print("[WARN] Rellenando temperaturas nulas con 25.0")
        df['temp_C'] = df['temp_C'].fillna(25.0)

    df_bal = balance_data_smart_2d(
        df, 
        limit_ambient=args.limit, 
        bin_size=args.bin_size, 
        seed=args.seed
    )
    
    df_bal.to_parquet(args.output, index=False)
    print(f"[Balance] Guardado en: {args.output}")

if __name__ == "__main__":
    main()