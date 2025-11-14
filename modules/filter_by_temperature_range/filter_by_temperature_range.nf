process filter_by_temperature_range {
  tag "filter_by_temperature_range"
  label 'cpu_small'
  publishDir path: { "${outdir}/curate" }, mode: 'copy', overwrite: true

  input:
    path source      
    val  outdir
    path filter_t_py
    val min
    val max

  output:
    path "filter_temp.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/curate"
  YAML="${baseDir}/envs/curate.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${filter_t_py}" --input "${source}" \
                                    --out filter_temp.parquet \
                                    --min-k "${min}" --max-k "${max}"

  """
}
