#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Train Full GBM
========================================================================================
    Description:
    Retrains gradient boosting models (XGBoost, LightGBM, CatBoost) on the full
    dataset using hyperparameters optimized during OOF training. These models
    are used for final inference.
    
    Models Trained:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Model     │ Output File  │ Additional Outputs                                 │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  XGBoost   │ xgb.pkl      │ shap_summary_xgb.png (feature importance)          │
    │  LightGBM  │ lgbm.pkl     │                                                     │
    │  CatBoost  │ cat.pkl      │                                                     │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Training Strategy:
    - Uses aggregated hyperparameters from K-fold tuning (median values)
    - Full dataset training (no validation split)
    - Sample weighting for temperature importance
    - GPU acceleration with automatic CPU fallback
    - SHAP analysis for model interpretability (XGBoost)
    
    Input:
    - Full training data with Mordred features
    - Hyperparameter directory from OOF training
    
    Output:
    - Trained model pickles for inference
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: TRAIN_FULL_GBM
========================================================================================
*/

process train_full_gbm {
    tag "Train Full GBM"
    label 'process_gpu'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training/${meta_id}/models_GBM", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(hp_dir)  // Training data and hyperparameters
        val  outdir_val                                      // Output directory
        path script_py                                       // Python script

    output:
        tuple val(meta_id), path("models_GBM"), emit: MODELS_DIR

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        def weight_col = params.full_gbm_weight_col ?: 'weight'
        def use_gpu = params.full_gbm_use_gpu != false ? '--use-gpu' : ''
        def weight_arg = weight_col ? "--sample-weight-col ${weight_col}" : ''
        
    """
    python ${script_py} \\
        --train "${train_file}" \\
        --target "logS" \\
        --hp-dir "${hp_dir}" \\
        ${use_gpu} \\
        ${weight_arg} \\
        --save-dir models_GBM
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
