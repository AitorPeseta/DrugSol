process outliers_scatter_mahalanobis {
  tag "outliers_scatter_mahalanobis"
  label 'cpu_small'
  publishDir path: { "${outdir}/analysis" }, mode: 'copy', overwrite: true

  input:
    tuple path(train), path(test)
    val  outdir
    path out_scatter_py

  output:
    path "out_viz", emit: OUTLIER_DIR
    
  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/analysis"
  YAML="${baseDir}/envs/analysis.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${out_scatter_py}" --train "${train}" --test "${test}" \
                                    --only_cols logS temperature_K \
                                    --exclude id target \
                                    --basis combined \
                                    --outlier_col is_outlier \
                                    --y-mode jitter --jitter-amp 0.12 --thresh 3.0 \
                                    --outdir out_viz
  """
}
