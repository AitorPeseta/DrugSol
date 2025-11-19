process train_full_tpsa {
  tag "train_full_tpsa"
  label 'cpu_small'
  publishDir path: { "${outdir}/training" }, mode: 'copy', overwrite: true

  input:
    path train
    val  outdir
    path train_full_py

  output:
    path "models_TPSA/tpsa_model.json",               emit: TPSA_MODEL


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
                                    --tpsa-col TPSA \
                                    --phenol-col n_phenol \
                                    --save-dir models_TPSA \

  """
}
  