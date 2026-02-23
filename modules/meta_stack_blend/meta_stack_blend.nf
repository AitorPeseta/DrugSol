#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Meta Stack Blend
========================================================================================
    Description:
    Ensemble meta-learning module that combines Out-of-Fold (OOF) predictions from
    multiple base models using two strategies: Stacking and Blending.
    
    Ensemble Strategies:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Strategy   │ Method                    │ Description                          │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Stacking   │ Ridge Regression          │ Learns weights via regularized       │
    │             │                           │ linear regression on OOF predictions │
    │             │                           │                                       │
    │  Blending   │ Simplex Projection        │ Non-negative weights summing to 1    │
    │             │                           │ via constrained optimization         │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Base Models Combined:
    - XGBoost (gradient boosting)
    - LightGBM (gradient boosting)
    - CatBoost (gradient boosting)
    - Chemprop (graph neural network)
    - Physics baseline (Ridge regression)
    
    The module automatically selects the best strategy based on cross-validated RMSE.
    
    Input:
    - OOF prediction files from each base model
    
    Output:
    - Stacking meta-model (Ridge)
    - Blending weights
    - Combined OOF predictions
    - Performance metrics
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: META_STACK_BLEND
========================================================================================
*/

process meta_stack_blend {
    tag "Ensemble Stacking & Blending"
    label 'cpu_small'
    
    conda "${params.conda_env_train}"
    
    publishDir "${params.outdir}/training/${meta_id}/ensemble", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(oof_xgb), path(oof_lgbm), path(oof_cat), path(oof_gnn), path(oof_physics)
        val  outdir_val
        path script_py

    output:
        tuple val(meta_id), path("meta_results/blend/weights.json"),      emit: BLEND_WEIGHTS
        tuple val(meta_id), path("meta_results/stack/meta_ridge.pkl"),    emit: STACK_MODEL
        tuple val(meta_id), path("meta_results/oof_predictions.parquet"), emit: OOF_COMBINED
        tuple val(meta_id), path("meta_results/metrics_oof.json"),        emit: METRICS

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        // Metric for model selection
        def metric = params.meta_metric ?: 'rmse'
        
        // Model labels (must match order of input files)
        def labels = params.meta_labels ?: 'xgb lgbm cat gnn physics'
        
    """
    python ${script_py} \\
        --oof-common "${oof_xgb}" "${oof_lgbm}" "${oof_cat}" "${oof_gnn}" "${oof_physics}" \\
        --labels ${labels} \\
        --metric ${metric} \\
        --save-dir meta_results
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
