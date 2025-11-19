process train_oof_tpsa {
  tag "train_oof_tpsa"
  label 'cpu_small'
  publishDir path: { "${outdir}/training" }, mode: 'copy', overwrite: true

  input:
    path train
    val  outdir
    path train_o_tpsa_py
    path folds

  output:
    path "oof_tpsa/oof_tpsa.parquet", emit: OFF_TPSA
    path "oof_tpsa/tpsa_oof_manifest.json", emit: MANI_GNN
    path "oof_tpsa/metrics_oof_tpsa.json", emit: META_GNN

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/train_methods"
  YAML="${baseDir}/envs/train_methods.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    rm -rf "\$PREFIX"
    ${params.MAMBA} clean --all -y
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority --always-copy
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${train_o_tpsa_py}" \
      --train "${train}" \
      --folds-file "${folds}" \
      --id-col row_uid \
      --target logS \
      --save-dir oof_tpsa
  """
}
