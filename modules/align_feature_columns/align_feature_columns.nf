process align_feature_columns {
  tag "align_feature_columns"
  label 'cpu_small'
  publishDir path: { "${outdir}/prepare_data" }, mode: 'copy', overwrite: true

  input:
    tuple path(train), path(test)
    val  outdir
    path align_py

  output:
    path "features_test_aligned.parquet", emit: out

  script:
  """
  set -euo pipefail
  set -x

  PREFIX="\$HOME/.conda_nf/prepare_data"
  YAML="${baseDir}/envs/prepare_data.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${align_py}" \
    "${train}" \
    "${test}" \
    "features_test_aligned.parquet"
  """
}
