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
    
    # Verificar que existen
    if not all(col in df.columns for col in cols):
        print("Error: El archivo no tiene las columnas de pKa esperadas.")
        print(f"Columnas encontradas: {list(df.columns)}")
        return

    print("-" * 50)
    print(" ESTADÍSTICAS DE pKa (Acidez/Basicidad)")
    print("-" * 50)

    # --- 1. ÁCIDOS ---
    # pKa Bajo = Ácido Fuerte
    # pKa Alto = Ácido Débil
    min_acid = df['pka_acid_min'].min()
    max_acid = df['pka_acid_max'].max()
    
    print(f"🔵 ÁCIDOS:")
    print(f"   - Límite Inferior (Más Fuerte): {min_acid:.2f}")
    print(f"   - Límite Superior (Más Débil):  {max_acid:.2f}")
    print(f"   * Nota: Si ves 16.0, es el valor por defecto para 'No Ácido'.")

    # --- 2. BASES ---
    # pKa Bajo = Base Débil (su ácido conjugado es fuerte)
    # pKa Alto = Base Fuerte (su ácido conjugado es débil)
    min_base = df['pka_base_min'].min()
    max_base = df['pka_base_max'].max()

    print(f"\n🔴 BASES (pKa del ácido conjugado):")
    print(f"   - Límite Inferior (Más Débil):  {min_base:.2f}")
    print(f"   - Límite Superior (Más Fuerte): {max_base:.2f}")
    print(f"   * Nota: Si ves -2.0, es el valor por defecto para 'No Base'.")
    print("-" * 50)

if __name__ == "__main__":
    main()