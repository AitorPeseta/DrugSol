#!/usr/bin/env python3
import argparse, os, pandas as pd

def read_any(p):
    pl = p.lower()
    if pl.endswith(".parquet"): return pd.read_parquet(p)
    return pd.read_csv(p)

def write_out(df, out_path, export_csv):
    tmp = out_path + ".part"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    if export_csv:
        csvp = os.path.splitext(out_path)[0] + ".csv"
        tmpc = csvp + ".part"
        df.to_csv(tmpc, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(tmpc, csvp)

def main():
    ap = argparse.ArgumentParser(
        description="Marca con is_drug=1 si aparece en ChEMBL (match por InChIKey14 o SMILES); 0 en caso contrario."
    )
    ap.add_argument("--in", dest="inp", required=True,
                    help="BigSolDB (parquet/csv) con smiles_neutral y/o InChIKey14")
    ap.add_argument("--chembl", required=True,
                    help="ChEMBL (parquet/csv) con InChIKey14 o canonical_smiles/smiles_neutral")
    ap.add_argument("--out", required=True,
                    help="Salida parquet del BigSolDB original + columna is_drug")
    ap.add_argument("--export-csv", action="store_true")
    args = ap.parse_args()

    df = read_any(args.inp)
    cm = read_any(args.chembl)

    # Normaliza columnas clave en BigSol
    if "InChIKey14" in df.columns:
        df["InChIKey14"] = df["InChIKey14"].astype("string")
    if "smiles_neutral" in df.columns:
        df["smiles_neutral"] = df["smiles_neutral"].astype("string")

    # Normaliza columnas clave en ChEMBL
    if "canonical_smiles" in cm.columns and "smiles_neutral" not in cm.columns:
        cm = cm.rename(columns={"canonical_smiles": "smiles_neutral"})
    if "InChIKey14" in cm.columns:
        cm["InChIKey14"] = cm["InChIKey14"].astype("string")
    if "smiles_neutral" in cm.columns:
        cm["smiles_neutral"] = cm["smiles_neutral"].astype("string")

    # Verificación de que al menos una clave existe en cada dataset
    have_key_df = ("InChIKey14" in df.columns) or ("smiles_neutral" in df.columns)
    have_key_cm = ("InChIKey14" in cm.columns) or ("smiles_neutral" in cm.columns)
    if not (have_key_df and have_key_cm):
        raise SystemExit(
            "Se requiere al menos una clave coincidente: InChIKey14 o smiles_neutral/canonical_smiles en ambos ficheros."
        )

    # Calcula is_drug sin traer columnas extra
    out = df.copy()
    in_chembl = pd.Series(False, index=out.index)

    if "InChIKey14" in df.columns and "InChIKey14" in cm.columns:
        in_chembl = in_chembl | out["InChIKey14"].isin(cm["InChIKey14"])

    if "smiles_neutral" in df.columns and "smiles_neutral" in cm.columns:
        in_chembl = in_chembl | out["smiles_neutral"].isin(cm["smiles_neutral"])

    out["is_drug"] = in_chembl.astype(int)

    # Tipos limpios (opcional)
    for c in ["solvent", "smiles_neutral", "InChIKey14"]:
        if c in out.columns:
            out[c] = out[c].astype("string")

    write_out(out, args.out, args.export_csv)
    print(f"[flag] Filas: {len(df)} | is_drug=1: {int(out['is_drug'].sum())} | out: {args.out}")

if __name__ == "__main__":
    main()
