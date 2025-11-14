process meta_stack_blend {
  tag "meta_stack_blend"
  label 'cpu_small'
  publishDir path: { "${outdir}/training" }, mode: 'copy', overwrite: true

  input:
    tuple path(train_lgbm), path(train_xgb), path(train_gnn), path(train_tpsa)
    val  outdir
    path meta_s_b
    val suffix

  output:
    path "meta_results/blend/weights${suffix}.json", emit: BLEND_WEIGHTS
    path "meta_results/stack/meta_ridge${suffix}.pkl", emit: STACK_MODEL
    path "meta_results/oof_predictions${suffix}.parquet", emit: OOF_COMBINED
    path "meta_results/metrics_oof${suffix}.json", emit: METRICS_OOF 
  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/train_methods"
  YAML="${baseDir}/envs/train_methods.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${meta_s_b}" \
                                    --oof-common \
                                      "${train_lgbm}" \
                                      "${train_xgb}" \
                                      "${train_gnn}" \
                                      "${train_tpsa}" \
                                    --labels lgbm xgb gnn tpsa \
                                    --metric rmse \
                                    --suffix ${suffix} \
                                    --save-dir meta_results

  """
}