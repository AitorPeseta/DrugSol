process train_oof_chemprop {
  tag "train_oof_chemprop"
  label 'cpu_small'
  publishDir path: { "${outdir}/training" }, mode: 'copy', overwrite: true

  input:
    path train
    val  outdir
    path train_o_gnn_py
    path folds

  output:
    path "oof_gnn/chemprop.parquet", emit: OFF_GNN
    path "oof_gnn/chemprop_best_params.json", emit: BEST_GNN
    path "oof_gnn/metrics_oof_chemprop.json", emit: META_GNN

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/train_methods"
  YAML="${baseDir}/envs/train_methods.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    rm -rf  "\$PREFIX"
    ${params.MAMBA} clean --all -y
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority --always-copy
  fi

  # Ejecución directa (Bypass micromamba run)
  "\$PREFIX/bin/python" "${train_o_gnn_py}" --train "${train}" --folds "${folds}" \\
                                    --smiles-col smiles_neutral \\
                                    --id-col row_uid \\
                                    --target logS --tune-trials 20 --epochs 40 \\
                                    --tune-pruner asha --asha-rungs 5 10 15\\
                                    --gpu --save-dir ./oof_gnn \\
  """
}