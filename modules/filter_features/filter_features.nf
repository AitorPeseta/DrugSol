process filter_features {
  tag "filter_features"
  label 'cpu_small'
  publishDir path: { "${outdir}/prepare_data" }, mode: 'copy', overwrite: true

  input:
    path file
    val  outdir
    path features_py
    val  name_out

  output:
    path "${name_out}_features_mordred_filtered.parquet", emit: out
    
  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/prepare_data"
  YAML="${baseDir}/envs/prepare_data.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  # Ejecución directa (Bypass micromamba run)
  "\$PREFIX/bin/python" "${features_py}" \\
                                    -i "${file}" \\
                                    -o ${name_out}_features_mordred_filtered.parquet \\
                                    --target logS

  """
}