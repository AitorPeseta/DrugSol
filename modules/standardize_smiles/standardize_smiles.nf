nextflow.enable.dsl = 2

process standardize_smiles {
  tag "standardize"
  label 'cpu_small'
  publishDir path: { "${outdir}/curate" }, mode: 'copy', overwrite: true

  input:
    path curated_parquet
    val  outdir
    path std_py

  output:
    path "standardize.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/curate"
  YAML="${baseDir}/envs/curate.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" \
    python ${std_py} --in ${curated_parquet} --out standardize.parquet --export-csv
  """
}
