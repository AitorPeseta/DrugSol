nextflow.enable.dsl = 2

process train_oof_gbm {
    tag "Train GBM (OOF)"
    label 'process_gpu'
    accelerator 1, type: 'nvidia' // Explicitly ask for NVIDIA GPU
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training", mode: 'copy', overwrite: true

    input:
        path train_file   // Parquet file with training data
        val  outdir_val
        path script_py    // Python script to train GBM models
        path folds_file   // Parquet file with fold assignments

    output:
        path "oof_gbm/oof/xgb.parquet",   emit: OFF_XGB
        path "oof_gbm/oof/lgbm.parquet",  emit: OFF_LGBM
        path "oof_gbm/metrics_tree.json", emit: METRICS_CH
        path "oof_gbm/hp",                emit: HP_DIR

    script:
    """
    # Train XGBoost & LightGBM with Out-of-Fold (OOF) predictions
    # Uses GPU acceleration and Optuna for hyperparameter tuning
    
    python ${script_py} \\
        --train "${train_file}" \\
        --folds "${folds_file}" \\
        --target logS \\
        --id-col row_uid \\
        --sample-weight-col sw_temp37 \\
        --use-gpu \\
        --tune-trials 15 \\
        --inner-splits 2 \\
        --pruner asha \\
        --asha-min-resource 1 \\
        --asha-reduction-factor 3 \\
        --asha-min-early-stopping-rate 0 \\
        --save-dir ./oof_gbm
    """
}