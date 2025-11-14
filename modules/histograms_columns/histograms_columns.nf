process histograms_columns {
  tag "histograms_columns"
  label 'cpu_small'
  publishDir path: { "${outdir}/analysis" }, mode: 'copy', overwrite: true

  input:
    tuple path(train), path(test)
    val  outdir
    path histogram_py

  output:
    path "hist_out", emit: HIST_DIR
    
  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/analysis"
  YAML="${baseDir}/envs/analysis.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${histogram_py}" --train "${train}" --test "${test}" \
                                    --cols logS temperature_K is_drug mordred__MW \
                                    --round-step 0.5 \
                                    --outdir hist_out
  """
}
