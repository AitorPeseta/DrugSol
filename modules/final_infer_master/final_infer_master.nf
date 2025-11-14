process final_infer_master {
  tag "final_infer_master"
  label 'cpu_small'
  publishDir path: { "${outdir}/pred" }, mode: 'copy', overwrite: true

  input:
    path test_tabular
    path test_smiles
    path models_dir
    path chemprop_dir
    path tpsa_model
    val  outdir
    path fim_py
    val  weights_json       // puede venir como null
    val  stack_model        // puede venir como null

  output:
    path "pred/test_level0.parquet"
    path "pred/test_blend.parquet"
    path "pred/test_stack.parquet", optional: true
    path "pred/metrics_test.json"

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/train_methods"
  YAML="${baseDir}/envs/train_methods.yml"
  [[ -d "\$PREFIX" ]] || ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority

  # ----- construir flags de forma segura en BASH -----
  WEIGHTS_OPT=""
  if [[ "${weights_json}" != "null" && -n "${weights_json}" && -f "${weights_json}" ]]; then
    WEIGHTS_OPT="--weights-json ${weights_json}"
  fi

  # dentro del script: del process final_infer_master
  STACK_OPT=""
  if [[ "${stack_model}" != "null" && -n "${stack_model}" && -f "${stack_model}" ]]; then
    STACK_OPT="--stack-pkl ${stack_model}"
  fi


  ${params.MAMBA} run -p "\$PREFIX" python "${fim_py}" \
      --test-tabular "${test_tabular}" \
      --test-smiles  "${test_smiles}"  \
      --models-dir   "${models_dir}"   \
      --id-col row_uid \
      --chemprop-model-dir "${chemprop_dir}" \
      --chemprop-smiles-col smiles \
      --tpsa-json "${tpsa_model}" --tpsa-col TPSA --phenol-col phenol_count \
      --smiles-col smiles_neutral --target logS \
      --save-dir pred \$WEIGHTS_OPT \$STACK_OPT
  """
}
