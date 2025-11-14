process fetch_chembl {
  tag "ChEMBL"
  label 'cpu_small'
  publishDir path: { "${outdir}/ingest" }, mode: 'copy', overwrite: true

  input:
    val  outdir
    path dl_py

  output:
    path "chembl_drugs.csv", emit: out

  script:
  """
  set -euo pipefail

  ENV_PREFIX="\$HOME/.conda_nf/common_ingest"
  if [[ ! -d "\$ENV_PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$ENV_PREFIX" -f ${baseDir}/envs/common_ingest.yml
  fi

  ${params.MAMBA} run -p "\$ENV_PREFIX" \\
    python ${dl_py} --out chembl_drugs.csv --max 3000 --pagesize 1000 --sleep 0.2 --only-approved

  # normaliza CSV (SIN línea en blanco y SIN sangría)
  ${params.MAMBA} run -p "\$ENV_PREFIX" \\
  python - <<'PY'
  import pandas as pd, os, sys
  f = "chembl_drugs.csv"
  try:
      df = pd.read_csv(f)
  except Exception as e:
      print("[ChEMBL] ERROR leyendo CSV:", e, file=sys.stderr); sys.exit(1)
  if df.shape[0] == 0:
      print("[ChEMBL] ERROR: CSV sin filas", file=sys.stderr); sys.exit(2)
  tmp = f + ".norm"
  df.to_csv(tmp, index=False, lineterminator="\\n", encoding="utf-8")
  os.replace(tmp, f)
  print(f"[ChEMBL] OK rows={len(df)}")

  df = pd.read_csv("chembl_drugs.csv", nrows=3)
  print("[fetch_chembl] cols:", list(df.columns))
  print(df[["canonical_smiles","max_phase","therapeutic_flag","atc_classifications"]].head())
  PY
  """
}
