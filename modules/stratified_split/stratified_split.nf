process stratified_split {
  tag "stratified_split"
  label 'cpu_small'
  publishDir path: { "${outdir}/prepare_data" }, mode: 'copy', overwrite: true

  input:
    path source      
    val  outdir
    path strat_py

  output:
    path "train.parquet", emit: train
    path "test.parquet", emit: test

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/prepare_data"
  YAML="${baseDir}/envs/prepare_data.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${strat_py}" --input "${source}" \
                                    --group-col cluster_ecfp4_0p7 \
                                    --temp-col temperature_K \
                                    --temp-step 2 \
                                    --test-size 0.2 \
                                    --seed 42 \
                                    --min-groups-per-class 2
  """
}
