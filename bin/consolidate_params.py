#!/usr/bin/env python3
import json
import sys
import os
import glob
import numpy as np
from collections import Counter
import argparse

def main():
    parser = argparse.ArgumentParser(description="Consolidar Hiperparámetros de múltiples folds")
    parser.add_argument('input_files', nargs='+', help='Lista de archivos o directorios de entrada')
    parser.add_argument('--output', default='best_params_consolidated.json', help='Nombre del archivo de salida')
    args = parser.parse_args()

    all_params = {}

    print(f"Procesando {len(args.input_files)} inputs...")

    # 1. Leer todos los inputs
    for f in args.input_files:
        target_file = f
        
        # Lógica inteligente: Si es directorio, buscar el JSON dentro
        if os.path.isdir(f):
            # Busca cualquier archivo que termine en .json dentro de la carpeta
            candidates = glob.glob(os.path.join(f, "*.json"))
            if not candidates:
                print(f"WARNING: No se encontró ningún .json en el directorio {f}. Saltando.")
                continue
            # Tomamos el primero que encuentre (usualmente solo hay uno: best_params.json)
            target_file = candidates[0]
            print(f" -> Directorio detectado. Usando archivo: {target_file}")

        # Leer el archivo
        try:
            with open(target_file, 'r') as fp:
                d = json.load(fp)
                for k, v in d.items():
                    if k not in all_params: all_params[k] = []
                    all_params[k].append(v)
        except Exception as e:
            print(f"ERROR leyendo {target_file}: {e}")

    if not all_params:
        print("ERROR: No se pudieron cargar parámetros. Revisa los inputs.")
        sys.exit(1)

    final_params = {}

    # 2. Consolidar (Promedio o Moda)
    for k, v_list in all_params.items():
        # Verificamos si es numérico y NO booleano
        if isinstance(v_list[0], (int, float)) and not isinstance(v_list[0], bool):
            mean_val = np.mean(v_list)
            # Si todos eran enteros puros, redondeamos a entero
            if all(isinstance(x, int) for x in v_list):
                final_params[k] = int(round(mean_val))
            else:
                final_params[k] = float(mean_val)
        else:
            # Para strings o booleanos, usamos la Moda
            counts = Counter(v_list)
            final_params[k] = counts.most_common(1)[0][0]

    # 3. Guardar
    with open(args.output, 'w') as fp:
        json.dump(final_params, fp, indent=4)
    
    print(f"Consolidado guardado exitosamente en {args.output}")

if __name__ == "__main__":
    main()