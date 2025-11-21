nextflow.enable.dsl = 2

process fetch_bigsoldb {
  tag "BigSolDB"
  label 'cpu_small'
  publishDir path: { "${outdir}/ingest" }, mode: 'copy', overwrite: true

  input:
    val  outdir
    val  record_id
    path dl_py

  output:
    path "bigsoldb.csv", emit: out

  script:
  """
  set -euo pipefail
  PREFIX="\$HOME/.conda_nf/common_ingest"
  YAML="${baseDir}/envs/common_ingest.yml"

  # 1. Creación del entorno (se mantiene igual)
  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  # 2. EJECUCIÓN DIRECTA: Descarga
  "\$PREFIX/bin/python" ${dl_py} --record ${record_id} --kind main --out bigsoldb.csv

  # 3. EJECUCIÓN DIRECTA: Normalización
  # normaliza a CSV LF/UTF-8
  "\$PREFIX/bin/python" - <<'PY'
  import pandas as pd, os
  f="bigsoldb.csv"
  df=pd.read_csv(f)
  tmp=f+".norm"
  df.to_csv(tmp,index=False,encoding="utf-8",lineterminator="\\n")
  os.replace(tmp,f)
  print(f"[BigSolDB] OK rows={len(df)}")
  PY
  """
}