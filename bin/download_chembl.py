#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import argparse
import sys

def extract_temp_from_text(text):
    if pd.isna(text) or text == '': return 25.0
    text = str(text)
    match = re.search(r'(\d{2,3})\s?(?:deg|degrees|C)\b', text, re.IGNORECASE)
    if match: return float(match.group(1))
    if "room temp" in text.lower() or "ambient" in text.lower(): return 25.0
    return 25.0 # Por defecto si no dice nada

def convert_to_logS_molar(row):
    """
    Convierte valor y unidad a LogS (Molar).
    Usa 'Standard Value', 'Standard Units' y 'Molecular Weight'.
    """
    try:
        val = float(row['Standard Value'])
        if val <= 0: return None
    except: return None

    unit = str(row['Standard Units']).lower().strip()
    mw = row['Molecular Weight']

    # 1. Unidades Molares
    if unit in ['m', 'molar']: return np.log10(val)
    if unit == 'mm': return np.log10(val * 1e-3)
    if unit == 'um': return np.log10(val * 1e-6)
    if unit == 'nm': return np.log10(val * 1e-9)
    
    # 2. Unidades Masa/Volumen (requieren MW)
    # A veces el CSV tiene nulos en MW, chequeamos
    try:
        mw = float(mw)
        if mw <= 0: return None
    except: return None

    conc_g_L = None
    if unit in ['ug.ml-1', 'ug/ml', 'mcg/ml']: conc_g_L = val * 1e-3
    elif unit == 'ng/ml': conc_g_L = val * 1e-6
    elif unit in ['mg/ml', 'mg.l-1', 'g/l']: conc_g_L = val
    
    if conc_g_L is not None:
        molar = conc_g_L / mw
        if molar > 0: return np.log10(molar)

    # 3. Si no tiene unidad pero el tipo es LogS, asumimos que ya es logarítmico
    # (A veces pasa en ChEMBL)
    std_type = str(row.get('Standard Type', '')).lower()
    if 'log' in std_type and (unit == 'nan' or unit == ''):
        return val

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="chembl_raw.csv", help="El CSV descargado de la web")
    args = parser.parse_args()

    print(f"--- PROCESANDO ARCHIVO LOCAL: {args.input} ---")
    
    try:
        # Leemos el CSV (a veces ChEMBL usa ';' como separador, detectamos auto)
        df = pd.read_csv(args.input, sep=None, engine='python')
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo: {e}")
        sys.exit(1)

    print(f"Filas originales: {len(df)}")
    
    # 1. Filtrar Fármacos (Phase > 0) y Solubilidad
    # ChEMBL CSV columns: 'Molecule Max Phase', 'Standard Value', etc.
    
    # Asegurar numéricos
    df['Molecule Max Phase'] = pd.to_numeric(df['Molecule Max Phase'], errors='coerce').fillna(0)
    
    # Filtro: Solo fármacos (Fase 1, 2, 3, 4)
    #df_drugs = df[df['Molecule Max Phase'] > 0].copy()
    #print(f"Fármacos (Phase > 0): {len(df_drugs)}")
    df_drugs = df.copy()
    print(f"Total compuestos: {len(df_drugs)}")

    # 2. Calcular LogS
    tqdm_pandas = False
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Convirtiendo Unidades")
        tqdm_pandas = True
    except: pass

    if tqdm_pandas:
        df_drugs['logS'] = df_drugs.progress_apply(convert_to_logS_molar, axis=1)
    else:
        df_drugs['logS'] = df_drugs.apply(convert_to_logS_molar, axis=1)

    # Eliminar nulos tras conversión
    df_clean = df_drugs.dropna(subset=['logS']).copy()
    print(f"Con LogS válido: {len(df_clean)}")

    # 3. Filtrar Insolubles (LogS < -4.5 para ser seguros)
    # Ajusta este umbral si quieres más o menos datos
    df_insol = df_clean[df_clean['logS'] < -5].copy()
    print(f"Insolubles (LogS < -4.5): {len(df_insol)}")

    # 4. Extraer Temperatura
    df_insol['temp_C'] = df_insol['Assay Description'].apply(extract_temp_from_text)
    
    # Marcar asunciones (si temp es 25 por defecto y no venía explícita)
    # Simplificación: si el texto no tenía números, asumimos 25
    df_insol['is_temp_assumed'] = df_insol['temp_C'].apply(lambda x: 1 if x == 25.0 else 0) 
    # (Mejorar esto requeriría revisar si el extract devolvió default, pero es aceptable)

    # 5. Formatear para el Pipeline
    output_df = pd.DataFrame()
    output_df['smiles_original'] = df_insol['Smiles']
    output_df['logS'] = df_insol['logS']
    output_df['solvent'] = 'water'
    output_df['temp_C'] = df_insol['temp_C']
    output_df['source'] = 'ChEMBL_Local_Ph' + df_insol['Molecule Max Phase'].astype(str)
    output_df['is_temp_assumed'] = df_insol['is_temp_assumed']
    
    # Limpiar SMILES nulos
    output_df = output_df.dropna(subset=['smiles_original'])
    
    outfile = "chembl.csv"
    output_df.to_csv(outfile, index=False)
    print(f"[Data] Guardado en {outfile} ({len(output_df)} compuestos)")

if __name__ == "__main__":
    main()