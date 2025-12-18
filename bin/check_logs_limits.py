#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_logs_limits.py
--------------------
Analiza los límites de LogS filtrando por:
1. Temperatura (Rango biológico/ambiental).
2. Solvente (Para no mezclar datos de etanol/metanol con agua).
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Peso molecular promedio para estimar mg/L si no tenemos el real
AVG_MW = 300.0 

def format_solubility(logs):
    """Convierte logS (mol/L) a mg/L aproximado"""
    molar = 10**logs
    mg_L = molar * AVG_MW * 1000
    
    # Formato amigable
    if mg_L < 0.001:
        return f"{molar:.2e} M ({mg_L*1000:.4f} µg/L)"
    elif mg_L > 1000:
        return f"{molar:.2f} M ({mg_L/1000:.2f} g/L)"
    else:
        return f"{molar:.4f} M ({mg_L:.2f} mg/L)"

def main():
    parser = argparse.ArgumentParser(description="Chequea límites de LogS con filtros estrictos.")
    parser.add_argument("input_file", help="Ruta a unified.parquet")
    parser.add_argument("--min-temp", type=float, default=None, help="Temp mínima °C")
    parser.add_argument("--max-temp", type=float, default=None, help="Temp máxima °C")
    parser.add_argument("--solvent", type=str, default="water", help="Solvente a analizar (default: water)")
    parser.add_argument("--ignore-solvent", action="store_true", help="Desactiva el filtro de solvente")
    
    args = parser.parse_args()
    
    print(f"Leyendo {args.input_file}...")
    try:
        df = pd.read_parquet(args.input_file)
    except:
        df = pd.read_csv(args.input_file)
    
    if "logS" not in df.columns:
        print("ERROR: No se encuentra la columna 'logS'")
        return

    initial_len = len(df)
    print(f"[Datos] Total inicial: {initial_len} filas")

    # --- 1. FILTRO DE SOLVENTE ---
    if not args.ignore_solvent:
        if "solvent" in df.columns:
            # Normalizar a minúsculas y quitar espacios
            target_solv = args.solvent.lower().strip()
            
            # Filtro flexible (water, h2o, agua...)
            if target_solv in ["water", "agua", "h2o"]:
                valid_solvents = ["water", "h2o", "agua", "aq"]
            else:
                valid_solvents = [target_solv]

            mask_solv = df["solvent"].astype(str).str.lower().str.strip().isin(valid_solvents)
            df = df[mask_solv]
            print(f"[Filtro] Solvente ('{args.solvent}'): {len(df)} filas restantes.")
        else:
            print("[WARN] No hay columna 'solvent'. No se puede filtrar.")

    # --- 2. FILTRO DE TEMPERATURA ---
    if "temp_C" in df.columns:
        if args.min_temp is not None or args.max_temp is not None:
            t_min = args.min_temp if args.min_temp is not None else -999
            t_max = args.max_temp if args.max_temp is not None else 999
            
            # Borrar NaNs de temperatura para ser estrictos
            df = df.dropna(subset=["temp_C"])
            df = df[(df["temp_C"] >= t_min) & (df["temp_C"] <= t_max)]
            print(f"[Filtro] Temp [{t_min}, {t_max}]°C: {len(df)} filas restantes.")
    else:
        if args.min_temp or args.max_temp:
            print("[WARN] No hay columna 'temp_C'. Ignorando filtro de temperatura.")

    if len(df) == 0:
        print("\n[ERROR] No quedan datos después de los filtros.")
        return

    # --- 3. ANÁLISIS ---
    min_val = df["logS"].min()
    max_val = df["logS"].max()
    
    print("\n" + "="*60)
    print(f" ESTADÍSTICAS (Solvente: {args.solvent} | Temp: {args.min_temp}-{args.max_temp}°C)")
    print("="*60)
    print(f"Muestras Válidas: {len(df)}")
    print(f"Rango LogS:       {min_val:.2f}  a  {max_val:.2f}")
    print("-" * 60)
    
    # MÍNIMO
    idx_min = df["logS"].idxmin()
    row_min = df.loc[idx_min]
    print(f"🔻 MÍNIMO (Más insoluble):")
    print(f"   LogS:   {row_min['logS']:.4f}")
    print(f"   Solub.: {format_solubility(row_min['logS'])}")
    print(f"   Temp:   {row_min.get('temp_C', 'N/A')} °C")
    print(f"   Solv:   {row_min.get('solvent', 'N/A')}")
    print(f"   SMILES: {row_min.get('smiles_neutral', 'N/A')}")

    print("-" * 60)
    
    # CHALLENGE CHECK
    target = -6.64
    count_lower = len(df[df["logS"] < target])
    print(f"🔎 COBERTURA CHALLENGE (Objetivo < {target}):")
    if count_lower > 0:
        print(f"   ✅ EXCELENTE. Tienes {count_lower} compuestos más insolubles que el objetivo.")
        print(f"      Tu modelo ha visto datos en esta zona extrema.")
    else:
        print(f"   ⚠️ PELIGRO. Tu base de datos NO baja hasta {target} en estas condiciones.")
        print(f"      El modelo estará extrapolando a ciegas.")

if __name__ == "__main__":
    main()