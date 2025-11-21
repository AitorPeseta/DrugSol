#!/usr/bin/env python3
import argparse, os, sys
import pandas as pd
import numpy as np

NEEDED = [
    'smiles_original','smiles_neutral',
    'solvent','temp_C','logS',
    'is_outlier',
    'sw_temp37'
]

# candidatos por campo (en orden de preferencia)
CANDIDATES = {
    'smiles_original': ['smiles_original','canonical_smiles','SMILES','smiles','SMILES_Solute'],
    'smiles_neutral' : ['smiles_neutral'],
    'solvent'        : ['solvent','Solvent','Medium','SMILES_Solvent'],
    'temperature_K'  : ['temperature_K','Temperature_K','temperature','Temperature','temp_K','T_K','T'],
    'logS'           : ['logS','LogS','logS_molL','Solubility','Y','LogS(mol/L)'],
    'is_outlier'     : ['is_outlier','outlier'],
}

def read_table(path: str) -> pd.DataFrame:
    """Lee CSV/Parquet con pequeños fallbacks para CSVs problemáticos."""
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine='python', encoding_errors='ignore')

def pick_series(df: pd.DataFrame, names) -> pd.Series:
    for c in names:
        if c in df.columns:
            return df[c]
    return pd.Series([pd.NA] * len(df), index=df.index)

def coerce_bool_nullable(s: pd.Series) -> pd.Series:
    if s.dtype == 'boolean':
        return s
    def _conv(x):
        if pd.isna(x): return pd.NA
        t = str(x).strip().lower()
        if t in ('true','1','yes','y','t'):  return True
        if t in ('false','0','no','n','f'):  return False
        return pd.NA
    return s.map(_conv).astype('boolean')

def to_numeric_nullable(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')

def load_one(path: str) -> pd.DataFrame:
    raw = read_table(path)

    out = pd.DataFrame(index=raw.index)

    # SMILES / solvent
    out['smiles_original']  = pick_series(raw, CANDIDATES['smiles_original']).astype('string')
    out['smiles_neutral']   = pick_series(raw, CANDIDATES['smiles_neutral']).astype('string')
    out['solvent']          = pick_series(raw, CANDIDATES['solvent']).astype('string')

    # temperature_K -> temp_C
    tempK = to_numeric_nullable(pick_series(raw, CANDIDATES['temperature_K']))
    out['temp_C'] = (tempK - 273.15).round(2)
    # Base de 1.0 para todos + Bonus gaussiano de 2.0 para los cercanos a 37
    out["sw_temp37"] = 1.0 + 2.0 * np.exp(-((out["temp_C"] - 37.0) ** 2) / (2 * 8.0**2))


    # logS
    out['logS']       = to_numeric_nullable(pick_series(raw, CANDIDATES['logS']))
    out['is_outlier'] = coerce_bool_nullable(pick_series(raw, CANDIDATES['is_outlier']))

    # Garantiza todas las columnas NEEDED y su orden
    for col in NEEDED:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA]*len(out), index=out.index)
    out = out[NEEDED]

    # Tipos finales consistentes
    for col in ['smiles_original','smiles_neutral','solvent']:
        out[col] = out[col].astype('string')
    for col in ['temp_C','logS']:
        out[col] = to_numeric_nullable(out[col])

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', required=True, help='CSV/Parquet a unificar')
    ap.add_argument('--export-csv', action='store_true', help='Exporta unified.csv además de Parquet')
    args = ap.parse_args()

    parts = []
    for f in args.sources:
        try:
            dfp = load_one(f)
            parts.append(dfp)
        except Exception as e:
            print(f"[unify] WARN: fallo procesando {f}: {e}", file=sys.stderr)

    if not parts:
        pd.DataFrame(columns=NEEDED).to_parquet('unified.parquet', index=False)
        if args.export_csv:
            pd.DataFrame(columns=NEEDED).to_csv('unified.csv', index=False)
        print("[unify] OK: 0 filas -> unified.parquet")
        sys.exit(0)

    df = pd.concat(parts, axis=0, ignore_index=True)

    df.to_parquet('unified.parquet', index=False)
    if args.export_csv:
        df.to_csv('unified.csv', index=False, lineterminator='\n', encoding='utf-8')

if __name__ == '__main__':
    main()