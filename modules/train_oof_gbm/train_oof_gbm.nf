process train_oof_gbm {
  tag "train_oof_gbm"
  label 'cpu_small'
  publishDir path: { "${outdir}/training" }, mode: 'copy', overwrite: true

  input:
    path train
    val  outdir
    path train_o_gbm_py
    path folds

  output:
    path "oof_gbm/oof/xgb.parquet",   emit: OFF_XGB
    path "oof_gbm/oof/lgbm.parquet",  emit: OFF_LGBM
    path "oof_gbm/metrics_tree.json", emit: METRICS_CH
    path "oof_gbm/hp",                emit: HP_DIR


  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/train_methods"
  YAML="${baseDir}/envs/train_methods.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${train_o_gbm_py}" --train "${train}" --folds "${folds}" \
                                    --target logS \
                                    --target logS \
                                    --use-gpu \
                                    --id-col row_uid \
                                    --tune-trials 35 \
                                    --inner-splits 3 \
                                    --pruner asha --asha-min-resource 1 \
                                    --asha-reduction-factor 3 \
                                    --asha-min-early-stopping-rate 0 \
                                    --save-dir ./oof_gbm


  """
}
//                                     --tune-trials 125 \
