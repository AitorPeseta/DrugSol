#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Train OOF GBM
========================================================================================
    Description:
    Trains gradient boosting models (XGBoost, LightGBM, CatBoost) using out-of-fold
    (OOF) cross-validation. OOF predictions are used for ensemble stacking.
    
    Models Trained:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Model     │ Library   │ GPU Support │ Key Features                            │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  XGBoost   │ xgboost   │ CUDA        │ Regularization, histogram-based splits  │
    │  LightGBM  │ lightgbm  │ OpenCL      │ Leaf-wise growth, categorical support   │
    │  CatBoost  │ catboost  │ CUDA        │ Ordered boosting, native categoricals   │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Training Strategy:
    - Pre-computed K-fold splits ensure scaffold separation
    - Optional Optuna hyperparameter tuning with ASHA pruning
    - Sample weighting to emphasize physiological temperature data
    - GPU acceleration when available (with CPU fallback)
    
    Output Files:
    - oof/xgb.parquet: XGBoost out-of-fold predictions
    - oof/lgbm.parquet: LightGBM out-of-fold predictions
    - oof/cat.parquet: CatBoost out-of-fold predictions
    - hp/*.json: Best hyperparameters per fold
    - metrics_tree.json: Overall metrics for each model
    
    Input:
    - Training data with Mordred features
    - Pre-computed fold assignments
    
    Output:
    - OOF predictions for ensemble stacking
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: TRAIN_OOF_GBM
========================================================================================
*/

process train_oof_gbm {
    tag "Train GBM (OOF)"
    label 'process_gpu'
    
    conda "${params.conda_env_train}"
    
    publishDir "${params.outdir}/training/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(folds_file)  // Training data and folds
        val  outdir_val                                          // Output directory
        path script_py                                           // Python script: train_oof_gbm.py

    output:
        tuple val(meta_id), path("oof_gbm/oof/xgb.parquet"),   emit: OOF_XGB
        tuple val(meta_id), path("oof_gbm/oof/lgbm.parquet"),  emit: OOF_LGBM
        tuple val(meta_id), path("oof_gbm/oof/cat.parquet"),   emit: OOF_CAT
        tuple val(meta_id), path("oof_gbm/metrics_tree.json"), emit: METRICS
        tuple val(meta_id), path("oof_gbm/hp"),                emit: HP_DIR

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        // Sample weight column (set to empty string to disable)
        def weight_col = params.gbm_weight_col ?: 'weight'
        def weight_arg = weight_col ? "--sample-weight-col ${weight_col}" : ''
        
        // GPU acceleration
        def use_gpu = params.gbm_use_gpu != false ? '--gpu' : ''
        
        // Optuna tuning settings
        def tune_trials = params.gbm_tune_trials ?: 50
        def inner_splits = params.gbm_inner_splits ?: 3
        
        // ASHA pruner settings
        def pruner = params.gbm_pruner ?: 'asha'
        def asha_min_resource = params.gbm_asha_min_resource ?: 1
        def asha_reduction = params.gbm_asha_reduction_factor ?: 3
        def asha_early_stop = params.gbm_asha_early_stopping_rate ?: 0
        
    """
    python ${script_py} \\
        --train "${train_file}" \\
        --folds "${folds_file}" \\
        --target "logS" \\
        --id-col "row_uid" \\
        ${weight_arg} \\
        ${use_gpu} \\
        --tune-trials ${tune_trials} \\
        --inner-splits ${inner_splits} \\
        --pruner ${pruner} \\
        --asha-min-resource ${asha_min_resource} \\
        --asha-reduction-factor ${asha_reduction} \\
        --asha-min-early-stopping-rate ${asha_early_stop} \\
        --save-dir ./oof_gbm
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
