process train_full_gbm {
  tag "train_full_gbm"
  label 'cpu_small'
  publishDir path: { "${outdir}/training" }, mode: 'copy', overwrite: true

  input:
    path train
    val  outdir
    path train_full_py
    path hp_dir

  output:
    path "models_GBM",                   emit: MODELS_DIR


  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/train_methods"
  YAML="${baseDir}/envs/train_methods.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${train_full_py}" \
                                    --train "${train}" \
                                    --target logS \
                                    --hp-dir "${hp_dir}" \
                                    --use-gpu \
                                    --save-dir models_GBM
  """
}