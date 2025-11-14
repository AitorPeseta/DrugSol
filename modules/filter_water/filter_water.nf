process filter_water {
  tag "filter_water"
  label 'cpu_small'
  publishDir path: { "${outdir}/curate" }, mode: 'copy', overwrite: true

  input:
    path source      
    val  outdir
    path filter_py

  output:
    path "filter_water.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/curate"
  YAML="${baseDir}/envs/curate.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${filter_py}" "${source}"
  """
}
