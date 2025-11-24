nextflow.enable.dsl = 2

process train_full_gbm {
    tag "Train Full GBM"
    label 'process_gpu' 
    accelerator 1, type: 'nvidia' // Explicitly ask for NVIDIA GPU
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training/models_GBM", mode: 'copy', overwrite: true

    input:
        path train_file  // Training data file
        val  outdir_val
        path script_py   // Python script
        path hp_dir // Directory containing best params from OOF

    output:
        path ".", emit: MODELS_DIR // Publish the whole folder content

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
        --save-dir .
    """
}