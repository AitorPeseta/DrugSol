process detect_outliers {
  tag "detect_outliers"
  label 'cpu_small'
  publishDir path: { "${outdir}/curate" }, mode: 'copy', overwrite: true

  input:
    path source      
    val  outdir
    path outlier_py

  output:
    path "detect_outliers.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/curate"
  YAML="${baseDir}/envs/curate.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  # Ejecución directa (Bypass micromamba run)
  "\$PREFIX/bin/python" "${outlier_py}"  --input ${source} \\
                                    --out detect_outliers.parquet \\
                                    --binning width --bins 10 \\
                                    --z-method robust --z-thresh 3.0 \\
                                    --export-csv
  """
}