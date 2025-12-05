import pandas as pd
import os

# TRUCO: Obtiene la ruta absoluta de DONDE está este archivo .py
directorio_del_script = os.path.dirname(os.path.abspath(__file__))

# Une esa ruta con el nombre del archivo
archivo = os.path.join(directorio_del_script, '../work/f6/de76ceba0c93880892df418abc80f8/final_train_gbm.parquet')

try:
    df = pd.read_parquet(archivo)
    print(f"--- Primeras 2 filas de {archivo} ---")
    print(df.head(2))

except FileNotFoundError:
    print(f"Error: No encuentro el archivo en: {archivo}")
except Exception as e:
    print(f"Ocurrió un error: {e}")