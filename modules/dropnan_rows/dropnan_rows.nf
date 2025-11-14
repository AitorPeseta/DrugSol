process dropnan_rows {
  tag "dropnan_rows"
  label 'cpu_small'
  publishDir path: { "${outdir}/prepare_data" }, mode: 'copy', overwrite: true

  input:
    path source
    val  outdir
    path drop_py
    val  name_out
    val  mode        

  output:
    path "${name_out}_dropnan.parquet", emit: out

  script:
  """
  set -euo pipefail
  set -x

  PREFIX="\$HOME/.conda_nf/prepare_data"
  YAML="${baseDir}/envs/prepare_data.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${drop_py}" \
    --input "${source}" \
    --output "${name_out}_dropnan.parquet" \
    --save_csv \
    --mode "${mode}"
  """
}
