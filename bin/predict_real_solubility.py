#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_real_solubility.py
--------------------------
Calcula la Solubilidad Efectiva a pH 7.4 (Fisiológico) partiendo de la predicción del modelo.

Flujo Termodinámico Correcto:
1. INPUT: Predicción del modelo (Sw experimental).
2. API: Obtiene pKas.
3. INTERNO: Calcula pH_sat y S0 (Intrínseca del Zwitterión/Neutro) usando fórmulas corregidas.
4. OUTPUT: Solubilidad recaliculada al pH objetivo (7.4).
"""

import argparse
import pandas as pd
import numpy as np
import requests
import time
from scipy.optimize import brentq
from tqdm import tqdm

# --- Configuración ---
API_URL = 'http://xundrug.cn:5001/modules/upload0/'
API_TOKEN = 'O05DriqqQLlry9kmpCwms2IJLC0MuLQ7'
KW = 1.0e-14 # Constante disociación agua

def get_pka_from_api(smiles):
    """Consulta la API con reintentos."""
    if not smiles or pd.isna(smiles): return [], []
    
    for _ in range(3): 
        try:
            response = requests.post(
                url=API_URL, files={"Smiles": ("tmg", smiles)}, 
                headers={'token': API_TOKEN}, timeout=5
            )
            if response.status_code == 200:
                data = response.json().get('gen_datas', {})
                # Extraer listas de pKas y limpiar valores vacíos
                acids = [float(v) for v in data.get('Acid', {}).values() if v]
                bases = [float(v) for v in data.get('Base', {}).values() if v]
                return acids, bases
        except:
            time.sleep(1)
    return [], []

def net_charge(pH, acid_pkas, base_pkas, Sw):
    """Calcula la carga neta para encontrar el pH de saturación."""
    h_conc = 10**(-pH)
    if h_conc == 0: return 0
    oh_conc = KW / h_conc
    
    charge = h_conc - oh_conc 
    
    for pka in acid_pkas:
        den = 1.0 + 10**(pka - pH)
        charge -= Sw * (1.0 / den)
        
    for pka in base_pkas:
        den = 1.0 + 10**(pH - pka)
        charge += Sw * (1.0 / den)
        
    return charge

def get_neutral_fraction(pH, acid_pkas, base_pkas):
    """
    FIX CRÍTICO: Calcula la fracción de la especie dominante neutra.
    Soporta Zwitteriones (Baclofen) para evitar el error de -7.9.
    """
    if not acid_pkas and not base_pkas:
        return 1.0

    # Tomamos el pKa más relevante de cada grupo
    pka_acid = min(acid_pkas) if acid_pkas else None
    pka_base = max(base_pkas) if base_pkas else None

    # --- LÓGICA CORREGIDA ---
    
    # 1. Anfolitos (Tienen Ácido y Base) -> Ej: Baclofen, Ciprofloxacino
    if pka_acid is not None and pka_base is not None:
        # ZWITTERIÓN (Ácido < Base): La forma "neutra" es la dipolar (+/-)
        if pka_acid < pka_base:
            # Fórmula exacta para Zwitteriones
            denom = 1.0 + 10**(pka_acid - pH) + 10**(pH - pka_base)
            return 1.0 / denom
        # ANFOLITO NORMAL (Base < Ácido): La forma neutra es sin carga
        else:
            prob_acid_neutral = 1.0 / (1.0 + 10**(pH - pka_acid))
            prob_base_neutral = 1.0 / (1.0 + 10**(pka_base - pH))
            return prob_acid_neutral * prob_base_neutral

    # 2. Solo Ácido
    if pka_acid is not None:
        return 1.0 / (1.0 + 10**(pH - pka_acid))

    # 3. Solo Base
    if pka_base is not None:
        return 1.0 / (1.0 + 10**(pka_base - pH))

    return 1.0

def process_molecule(row, target_ph):
    """Calcula Seff a pH objetivo usando S0 como puente."""
    try:
        log_sw = float(row['predicted_logS'])
        sw_molar = 10**log_sw
        smi = row['smiles']
    except:
        return None

    acid_pkas, base_pkas = get_pka_from_api(smi)
    
    # Si no hay pKas, no afecta el pH
    if not acid_pkas and not base_pkas:
        return {
            'pka_acids': "[]", 'pka_bases': "[]",
            f'logSeff_pH{target_ph}': log_sw  # Se queda igual
        }

    # 1. Calcular pH de saturación (donde se midió Sw teóricamente)
    try:
        ph_sat = brentq(net_charge, -2, 16, args=(acid_pkas, base_pkas, sw_molar))
    except:
        ph_sat = 7.0

    # 2. Calcular S0 (Puente Matemático)
    # Usamos la función CORREGIDA para que los zwitteriones no den valores absurdos
    f_neutra_sat = get_neutral_fraction(ph_sat, acid_pkas, base_pkas)
    s0_molar = sw_molar * f_neutra_sat
    if s0_molar < 1e-15: s0_molar = 1e-15 # Protección numérica

    # 3. Calcular Seff al pH Objetivo (7.4)
    f_neutra_target = get_neutral_fraction(target_ph, acid_pkas, base_pkas)
    if f_neutra_target < 1e-12: f_neutra_target = 1e-12
    
    seff_molar = s0_molar / f_neutra_target
    log_seff = np.log10(seff_molar)

    return {
        'pka_acids': str(acid_pkas),
        'pka_bases': str(base_pkas),
        'pH_sat_calculated': round(ph_sat, 2),
        f'logSeff_pH{target_ph}': round(log_seff, 3)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV con 'predicted_logS'")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ph", type=float, default=7.4, help="pH objetivo (default 7.4)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"[Thermo] Calculando solubilidad a pH {args.ph} para {len(df)} moléculas...")

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        res = process_molecule(row, args.ph)
        results.append(res if res else {})

    res_df = pd.DataFrame(results)
    final_df = pd.concat([df, res_df], axis=1)

    # Ordenar columnas bonitas
    cols = list(final_df.columns)
    important = ['smiles', 'predicted_logS', f'logSeff_pH{args.ph}', 'pka_acids', 'pka_bases']
    for c in reversed(important):
        if c in cols:
            cols.insert(0, cols.pop(cols.index(c)))
    
    # Limpieza: Si quieres borrar la columna S0 para que no la vea tu jefe, 
    # simplemente no la incluimos en el DataFrame final (ya no está en 'results')
    
    final_df.to_csv(args.output, index=False)
    print(f"[Thermo] Hecho. Guardado en {args.output}")

if __name__ == "__main__":
    main()