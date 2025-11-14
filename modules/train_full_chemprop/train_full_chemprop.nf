process train_full_chemprop {
  tag "train_full_chemprop"
  label 'cpu_small'
  publishDir path: { "${outdir}/training" }, mode: 'copy', overwrite: true

  input:
    path train
    val  outdir
    path train_full_py
    path best_params

  output:
    path "models_GNN",               emit: CHEMPROP_DIR
    path "models_GNN/chemprop_manifest.json", emit: CHEMPROP_MANIFEST


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
                                    --smiles-col smiles_neutral \
                                    --target logS \
                                    --gpu \
                                    --epochs 40 \
                                    --best-params "${best_params}" \
                                    --save-dir models_GNN
  """
}
