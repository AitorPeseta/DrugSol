import sys
import pandas as pd

# Verificar que se haya pasado un argumento
if len(sys.argv) < 2:
    print("Uso: python filtrar_water.py <ruta_al_archivo.parquet>")
    sys.exit(1)

# Tomar el nombre del archivo desde los argumentos
input_file = sys.argv[1]

# Leer el archivo Parquet
df = pd.read_parquet(input_file)

# Filtrar solo las filas donde solvent == "water"
df_water = df[df["solvent"].str.lower() == "water"]

# Crear un nombre de salida (curated.parquet)
output_file = "filter_water.parquet"

# Guardar el archivo filtrado
df_water.to_parquet(output_file, index=False)
