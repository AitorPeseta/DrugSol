#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Final Inference Master
========================================================================================
    Description:
    Master inference pipeline that generates predictions from all trained models
    and combines them using learned ensemble weights.
    
    Inference Pipeline:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  1. Load test data (tabular features + SMILES)                                 │
    │  2. Generate Level-0 predictions:                                               │
    │     - XGBoost, LightGBM, CatBoost (from GBM models)                           │
    │     - Chemprop D-MPNN (from GNN model)                                         │
    │     - Physics baseline (from Ridge model)                                       │
    │  3. Combine predictions:                                                        │
    │     - Blending: Weighted average using learned weights                         │
    │     - Stacking: Meta-model prediction using Ridge coefficients                 │
    │  4. Calculate metrics (including physiological temperature range)              │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Input:
    - Test data (tabular features + SMILES)
    - Trained models (GBM, GNN, Physics)
    - Ensemble weights/meta-model
    
    Output:
    - Level-0 predictions from all base models
    - Blended predictions
    - Stacked predictions (optional)
    - Performance metrics
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: FINAL_INFER_MASTER
========================================================================================
*/

process final_infer_master {
    tag "Final Inference"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/evaluation/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(test_tab), path(test_smi), path(mod_gbm), path(mod_gnn), path(mod_physics), path(weights), path(stack)
        val  outdir
        path script_py

    output:
        tuple val(meta_id), path("pred/test_level0.parquet"), emit: LEVEL0
        tuple val(meta_id), path("pred/test_blend.parquet"),  emit: BLEND
        tuple val(meta_id), path("pred/test_stack.parquet"),  emit: STACK, optional: true
        tuple val(meta_id), path("pred/metrics_test.json"),   emit: METRICS

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        def id_col = params.infer_id_col ?: 'row_uid'
        def smiles_col = params.infer_smiles_col ?: 'smiles_neutral'
        def target_col = params.infer_target_col ?: 'logS'
        
    """
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
        --test-smiles "${test_smi}" \\
        --models-dir "${mod_gbm}" \\
        --chemprop-model-dir "${mod_gnn}" \\
        --physics-json "${mod_physics}" \\
        --id-col ${id_col} \\
        --smiles-col ${smiles_col} \\
        --target ${target_col} \\
        --save-dir pred \\
        \$WEIGHTS_OPT \$STACK_OPT
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
