process unify_datasets {
  tag "unify_datasets"
  label 'cpu_small'
  publishDir path: { "${outdir}/ingest" }, mode: 'copy', overwrite: true

  input:
    path source      
    val  outdir
    path unify_py

  output:
    path "unified.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/common_ingest"
  YAML="${baseDir}/envs/common_ingest.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${unify_py}" --sources "${source}" --export-csv
  """
}
