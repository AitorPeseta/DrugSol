import pandas as pd
import sys

# Ajusta el nombre del archivo si es necesario
INPUT_FILE = "../results/research/prepare_data/engineered_features.parquet"

def main():
    try:
        print(f"Cargando {INPUT_FILE}...")
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo {INPUT_FILE}")
        return

    # Columnas de pKa generadas por engineer_features.py
    cols = ['pka_acid_min', 'pka_acid_max', 'pka_base_min', 'pka_base_max']
    
    if not all(col in df.columns for col in cols):
        print("Error: El archivo no tiene las columnas de pKa esperadas.")
        print(f"Columnas encontradas: {list(df.columns)}")
        return

    total_mols = len(df)
    print("-" * 60)
    print(f" ESTADÍSTICAS DE pKa (Total Moléculas: {total_mols})")
    print("-" * 60)

    # --- 1. ÁCIDOS ---
    # Usamos 'pka_acid_min' porque representa el grupo MÁS ácido de la molécula.
    # Filtramos los que tienen 16.0, ya que esos son "No Ácidos" (valor por defecto).
    real_acids = df[df['pka_acid_min'] < 15.9]
    n_acids = len(real_acids)

    if n_acids > 0:
        min_acid = real_acids['pka_acid_min'].min()
        max_acid = real_acids['pka_acid_max'].max()
        
        # Conteos de umbrales
        gt_2 = real_acids[real_acids['pka_acid_min'] > 4].shape[0]
        gt_10 = real_acids[real_acids['pka_acid_min'] > 10].shape[0]

        print(f"🔵 ÁCIDOS DETECTADOS: {n_acids} ({n_acids/total_mols:.1%} del total)")
        print(f"   - Rango Real detectado: {min_acid:.2f} a {max_acid:.2f}")
        print(f"   - Superan pKa > 4 (Débiles):     {gt_2} ({gt_2/n_acids:.1%})")
        print(f"   - Superan pKa > 10 (Muy Débiles): {gt_10} ({gt_10/n_acids:.1%})")
        print(f"     *(Nota: Cuanto más alto el pKa, más débil es el ácido)*")
    else:
        print("🔵 ÁCIDOS: No se encontraron moléculas con grupos ácidos reales.")

    print("-" * 60)

    # --- 2. BASES ---
    # Usamos 'pka_base_max' porque representa el grupo MÁS básico de la molécula.
    # Filtramos los que tienen -2.0, ya que esos son "No Bases".
    real_bases = df[df['pka_base_max'] > -1.9]
    n_bases = len(real_bases)

    if n_bases > 0:
        min_base = real_bases['pka_base_min'].min()
        max_base = real_bases['pka_base_max'].max()

        # Conteos de umbrales
        # En bases, pKa > 2 ya es una base relevante.
        # pKa > 10 es una base fuerte.
        gt_2 = real_bases[real_bases['pka_base_max'] > 2].shape[0]
        gt_10 = real_bases[real_bases['pka_base_max'] > 10].shape[0]

        print(f"🔴 BASES DETECTADAS: {n_bases} ({n_bases/total_mols:.1%} del total)")
        print(f"   - Rango Real detectado: {min_base:.2f} a {max_base:.2f}")
        print(f"   - Superan pKa > 2 (Relevantes):  {gt_2} ({gt_2/n_bases:.1%})")
        print(f"   - Superan pKa > 10 (Fuertes):    {gt_10} ({gt_10/n_bases:.1%})")
        print(f"     *(Nota: Cuanto más alto el pKa conjugado, más fuerte es la base)*")
    else:
        print("🔴 BASES: No se encontraron moléculas con grupos básicos reales.")

    print("-" * 60)

if __name__ == "__main__":
    main()