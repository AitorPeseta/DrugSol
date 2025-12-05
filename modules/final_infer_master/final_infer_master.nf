nextflow.enable.dsl = 2

process final_infer_master {
    tag "Final Inference"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/evaluation/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(test_tab), path(test_smi), path(mod_gbm), path(mod_gnn), path(mod_tpsa), path(weights), path(stack)
        val outdir
        path script_py

    output:
        tuple val(meta_id), path("pred/test_level0.parquet"), emit: LEVEL0
        tuple val(meta_id), path("pred/test_blend.parquet"),  emit: BLEND
        tuple val(meta_id), path("pred/test_stack.parquet"),  emit: STACK, optional: true
        tuple val(meta_id), path("pred/metrics_test.json"),   emit: METRICS

    script:
    """
    # Final Inference Pipeline
    # 1. Generates predictions from all base models (XGB, LGBM, GNN, TPSA)
    # 2. Combines them using learned weights/stacking model
    # 3. Calculates final metrics (including physiological range)
    
    # Build optional flags
    WEIGHTS_OPT=""
    if [[ -f "${weights}" ]]; then
        WEIGHTS_OPT="--weights-json ${weights}"
    fi

    STACK_OPT=""
    if [[ -f "${stack}" ]]; then
        STACK_OPT="--stack-pkl ${stack}"
    fi
    
    python ${script_py} \\
        --test-tabular "${test_tab}" \\
        --test-smiles  "${test_smi}"  \\
        --models-dir   "${mod_gbm}"   \\
        --chemprop-model-dir "${mod_gnn}" \\
        --tpsa-json "${mod_tpsa}" \\
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