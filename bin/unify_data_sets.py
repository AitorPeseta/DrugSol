#!/usr/bin/env python3
import argparse, os, sys
import pandas as pd

NEEDED = [
    'smiles_original','smiles_neutral','InChIKey14',
    'solvent','temperature_K','logS',
    'is_outlier'
]

# candidatos por campo (en orden de preferencia)
CANDIDATES = {
    'smiles_original': ['smiles_original','canonical_smiles','SMILES','smiles','SMILES_Solute'],
    'smiles_neutral' : ['smiles_neutral'],
    'InChIKey14'     : ['InChIKey14','InChIKey','inchikey14','inchikey'],
    'solvent'        : ['solvent','Solvent','Medium','SMILES_Solvent'],
    'temperature_K'  : ['temperature_K','Temperature_K','temperature','Temperature','temp_K','T_K','T'],
    'logS'           : ['logS','LogS','logS_molL','Solubility','Y','LogS(mol/L)'],
    'is_outlier'      : ['is_outlier','outlier'],
}

def read_table(path: str) -> pd.DataFrame:
    """Lee CSV/Parquet con pequeños fallbacks para CSVs problemáticos."""
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    # CSV: intenta por defecto y luego con engine=python si hiciera falta
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

    # Mapeos SOLO a NEEDED (sin 'smiles')
    out['smiles_original']  = pick_series(raw, CANDIDATES['smiles_original']).astype('string')
    out['smiles_neutral']   = pick_series(raw, CANDIDATES['smiles_neutral']).astype('string')

    # InChIKey14 directo si viene completo o derivado si trae InChIKey
    inchikey = pick_series(raw, CANDIDATES['InChIKey14']).astype('string')
    # si es el InChIKey completo, recorta a 14
    out['InChIKey14'] = inchikey.str.slice(0, 14)

    out['solvent']          = pick_series(raw, CANDIDATES['solvent']).astype('string')
    out['temperature_K']    = to_numeric_nullable(pick_series(raw, CANDIDATES['temperature_K']))
    out['logS']             = to_numeric_nullable(pick_series(raw, CANDIDATES['logS']))

    out['is_outlier']       = coerce_bool_nullable(pick_series(raw, CANDIDATES['is_outlier']))

    # Garantiza todas las columnas NEEDED y su orden
    for col in NEEDED:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA]*len(out), index=out.index)
    out = out[NEEDED]

    # Tipos finales consistentes
    for col in ['smiles_original','smiles_neutral','InChIKey14','solvent']:
        out[col] = out[col].astype('string')
    for col in ['temperature_K','logS']:
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

    # Guarda
    df.to_parquet('unified.parquet', index=False)
    if args.export_csv:
        # CSV “amistoso” (LF y UTF-8)
        df.to_csv('unified.csv', index=False, lineterminator='\n', encoding='utf-8')

if __name__ == '__main__':
    main()
