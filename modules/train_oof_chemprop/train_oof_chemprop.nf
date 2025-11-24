nextflow.enable.dsl = 2

process train_oof_chemprop {
    tag "Train Chemprop (OOF)"
    label 'process_gpu'
    accelerator 1, type: 'nvidia'   // Explicitly ask for NVIDIA GPU
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training", mode: 'copy', overwrite: true

    input:
        path train_file   // Training data file
        val  outdir_val
        path script_py    // Python script
        path folds_file   // Cross-validation folds file

    output:
        path "oof_gnn/chemprop.parquet", emit: OFF_GNN
        path "oof_gnn/chemprop_best_params.json", emit: BEST_GNN
        path "oof_gnn/metrics_oof_chemprop.json", emit: META_GNN

    script:
    """
    # Train Chemprop GNN with OOF prediction
    # Uses GPU and Optuna
    
    python ${script_py} \\
        --train "${train_file}" \\
        --folds "${folds_file}" \\
        --smiles-col smiles_neutral \\
        --id-col row_uid \\
        --target logS \\
        --weight-col sw_temp37 \\
        --tune-trials 20 \\
        --epochs 40 \\
        --tune-pruner asha \\
        --asha-rungs 5 10 15 \\
        --gpu \\
        --save-dir ./oof_gnn
    """
}