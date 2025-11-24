nextflow.enable.dsl = 2

process final_infer_master {
    tag "Final Inference"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/prediction", mode: 'copy', overwrite: true

    input:
        path test_tabular  // Parquet file features
        path test_smiles   // Parquet file SMILES
        path models_dir    // Directory with base models
        path chemprop_dir  // Directory with Chemprop models
        path tpsa_model    // TPSA model JSON
        val  outdir_val
        path script_py     // Python script
        path weights_json  // Optional
        path stack_model   // Optional

    output:
        path "pred/test_level0.parquet", emit: LEVEL0
        path "pred/test_blend.parquet",  emit: BLEND
        path "pred/test_stack.parquet",  emit: STACK, optional: true
        path "pred/metrics_test.json",   emit: METRICS

    script:
    """
    # Final Inference Pipeline
    # 1. Generates predictions from all base models (XGB, LGBM, GNN, TPSA)
    # 2. Combines them using learned weights/stacking model
    # 3. Calculates final metrics (including physiological range)
    
    # Build optional flags
    WEIGHTS_OPT=""
    if [[ -f "${weights_json}" ]]; then
        WEIGHTS_OPT="--weights-json ${weights_json}"
    fi

    STACK_OPT=""
    if [[ -f "${stack_model}" ]]; then
        STACK_OPT="--stack-pkl ${stack_model}"
    fi
    
    python ${script_py} \\
        --test-tabular "${test_tabular}" \\
        --test-smiles  "${test_smiles}"  \\
        --models-dir   "${models_dir}"   \\
        --chemprop-model-dir "${chemprop_dir}" \\
        --tpsa-json "${tpsa_model}" \\
        --id-col row_uid \\
        --smiles-col smiles_neutral \\
        --chemprop-smiles-col smiles \\
        --tpsa-col TPSA \\
        --phenol-col n_phenol \\
        --target logS \\
        --save-dir pred \\
        \$WEIGHTS_OPT \$STACK_OPT
    """
}