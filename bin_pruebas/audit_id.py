#!/usr/bin/env python3
import sys, pandas as pd, numpy as np, re
from pathlib import Path

def peek(p):
    df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    n = min(5, len(df))
    print(f"\n== {p} ==")
    print(df.head(n).to_string(index=False))
    cols = set(df.columns)
    id_cols = [c for c in ["row_uid","InChIKey14","id","ID","Id"] if c in cols]
    if id_cols:
        c = id_cols[0]
        s = df[c].astype(str)
        # heurística: row_uid suele llevar pipes y ‘#’
        has_pipe = s.str.contains(r"\|").mean()
        has_hash = s.str.contains(r"#").mean()
        print(f"[info] id_col={c} | dtype={df[c].dtype} | '|'={has_pipe:.2f} | '#'={has_hash:.2f}")
    else:
        print("[warn] no id column found among {row_uid, InChIKey14, id}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        if Path(p).exists():
            peek(p)
        else:
            print(f"[skip] {p} (not found)")
