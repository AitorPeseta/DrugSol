#!/usr/bin/env python3
import pandas as pd
import argparse
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[Concat] Uniendo {args.train} y {args.test}...")
    
    try:
        df_tr = pd.read_parquet(args.train)
        df_te = pd.read_parquet(args.test)
        
        # Concatenar
        df_full = pd.concat([df_tr, df_te], ignore_index=True)
        
        # Eliminar columna 'fold' si existe (ya no tiene sentido en full)
        if "fold" in df_full.columns:
            df_full = df_full.drop(columns=["fold"])
            
        # Eliminar duplicados por si acaso (seguridad)
        if "row_uid" in df_full.columns:
            df_full = df_full.drop_duplicates(subset=["row_uid"])
            
        print(f"[Concat] Total filas: {len(df_full)}")
        df_full.to_parquet(args.out, index=False)
        
    except Exception as e:
        print(f"[ERROR] Fallo al concatenar: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()