nextflow.enable.dsl = 2

process train_oof_gbm {
    tag "Train GBM (OOF)"
    label 'process_gpu'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(folds_file)
        val  outdir_val
        path script_py    // Python script to train GBM models

    output:
        tuple val(meta_id), path("oof_gbm/oof/xgb.parquet"),   emit: OFF_XGB
        tuple val(meta_id), path("oof_gbm/oof/lgbm.parquet"),  emit: OFF_LGBM
        tuple val(meta_id), path("oof_gbm/metrics_tree.json"), emit: METRICS_CH
        tuple val(meta_id), path("oof_gbm/hp"),                emit: HP_DIR

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
        --tune-trials 5 \\
        --inner-splits 2 \\
        --pruner asha \\
        --asha-min-resource 1 \\
        --asha-reduction-factor 3 \\
        --asha-min-early-stopping-rate 0 \\
        --save-dir ./oof_gbm
    """
}