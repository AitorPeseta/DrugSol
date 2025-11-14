import sys
import os
import numpy as np
import pandas as pd

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Extensión no soportada: {ext} (usa .parquet o .csv)")

def write_any(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Extensión no soportada: {ext} (usa .parquet o .csv)")

def main():
    if len(sys.argv) < 4:
        print("Uso: python align_feature_columns.py <features_train.(parquet|csv)> <features_test.(parquet|csv)> <features_test_aligned.(parquet|csv)>")
        sys.exit(1)

    train_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    tr = read_any(train_path)
    te = read_any(test_path)

    cols_tr = tr.columns.tolist()
    cols_te = te.columns.tolist()

    # Añadir a test todas las columnas que existan en train y falten en test
    missing_in_test = [c for c in cols_tr if c not in cols_te]
    for c in missing_in_test:
        if c.startswith("solv_"):
            te[c] = 0
        else:
            te[c] = np.nan

    # Eliminar de test las columnas que no existen en train
    extra_in_test = [c for c in cols_te if c not in cols_tr]
    if extra_in_test:
        te = te.drop(columns=extra_in_test, errors="ignore")

    # Reordenar como train (clave para que el Pipeline matchee columnas)
    te = te[cols_tr]

    # Normalizar 0/1 para 'solv_' por si llegan como bool/str
    for c in te.columns:
        if c.startswith("solv_"):
            te[c] = (te[c]
                     .map({True:1, False:0, "True":1, "False":0, "true":1, "false":0})
                     .fillna(te[c])
                     .astype("int8"))

    write_any(te, out_path)
    print("Test alineado guardado en:", out_path)
    print(f"Faltaban en test: {len(missing_in_test)} | Sobraban en test: {len(extra_in_test)}")
    if missing_in_test:
        print("   Añadidas (primeras 10):", missing_in_test[:10])
    if extra_in_test:
        print("   Eliminadas (primeras 10):", extra_in_test[:10])

if __name__ == "__main__":
    main()
