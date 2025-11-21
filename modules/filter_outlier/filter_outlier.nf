process filter_outlier {
  tag "filter_outlier"
  label 'cpu_small'
  publishDir path: { "${outdir}/curate" }, mode: 'copy', overwrite: true

  input:
    path source      
    val  outdir
    path outlier_py

  output:
    path "filter_outlier.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/curate"
  YAML="${baseDir}/envs/curate.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  # Ejecución directa (Bypass micromamba run)
  "\$PREFIX/bin/python" "${outlier_py}" -i "${source}" --save_csv
  """
}