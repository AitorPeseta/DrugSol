nextflow.enable.dsl = 2

process train_full_gbm {
    tag "Train Full GBM"
    label 'process_gpu' 
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training/${meta_id}/models_GBM", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(hp_dir)
        val  outdir_val
        path script_py   // Python script

    output:
        tuple val(meta_id), path("models_GBM"), emit: MODELS_DIR // Publish the whole folder content

    script:
    """
    # Retrain XGBoost & LightGBM on 100% data using best hyperparameters
    # Uses GPU if available
    
    python ${script_py} \\
        --train "${train_file}" \\
        --target logS \\
        --hp-dir "${hp_dir}" \\
        --use-gpu \\
        --sample-weight-col sw_temp37 \\
        --save-dir models_GBM
    """
}