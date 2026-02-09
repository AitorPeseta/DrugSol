#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Train OOF Chemprop
========================================================================================
    Description:
    Trains a Chemprop graph neural network (D-MPNN) for molecular property prediction.
    Uses only SMILES structure for learning, without external descriptors.
    
    Training Strategy:
    - Pure graph approach: uses only SMILES structure, no external features
    - Optional Optuna hyperparameter tuning with early stopping
    - Sample weighting via data_weights_path
    - GPU acceleration for faster training
    
    Hyperparameters Tuned:
    - message_hidden_dim: Hidden size of message passing layers (300-800)
    - depth: Number of message passing iterations (2-5)
    - dropout: Dropout rate (0.0-0.3)
    - ffn_num_layers: Number of FFN layers (1-2)
    
    Input:
    - Training data with SMILES column
    - Pre-computed fold assignments
    
    Output:
    - OOF predictions for ensemble stacking
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: TRAIN_OOF_CHEMPROP
========================================================================================
*/

process train_oof_chemprop {
    tag "Train Chemprop (OOF)"
    label 'process_gpu'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(folds_file)  // Training data and folds
        val  outdir_val                                          // Output directory
        path script_py                                           // Python script

    output:
        tuple val(meta_id), path("oof_gnn/chemprop.parquet"),           emit: OOF_GNN
        tuple val(meta_id), path("oof_gnn/chemprop_best_params.json"),  emit: BEST_PARAMS
        tuple val(meta_id), path("oof_gnn/metrics_oof_chemprop.json"),  emit: METRICS

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        // Column names
        def weight_col = params.chemprop_weight_col ?: 'weight'
        
        // Training settings
        def epochs = params.chemprop_epochs ?: 50
        def batch_size = params.chemprop_batch_size ?: 32
        
        // Optuna tuning
        def tune_trials = params.chemprop_tune_trials ?: 20
        def tune_pruner = params.chemprop_pruner ?: 'asha'
        def asha_rungs = params.chemprop_asha_rungs ?: 5
        
    """
    set -euo pipefail

    # MKL library preload for compatibility
    echo "Setting up MKL libraries from: \$CONDA_PREFIX"
    
    export LD_PRELOAD=\$(find "\$CONDA_PREFIX/lib" -name "libmkl_core.so*" 2>/dev/null | head -n 1)
    
    if [ -z "\$LD_PRELOAD" ]; then
        export LD_PRELOAD=\$(find "\$CONDA_PREFIX/lib" -name "libmkl_intel_lp64.so*" 2>/dev/null | head -n 1)
    fi
    
    if [ -n "\$LD_PRELOAD" ]; then
        echo "LD_PRELOAD set to: \$LD_PRELOAD"
    fi
    
    python ${script_py} \\
        --train "${train_file}" \\
        --folds "${folds_file}" \\
        --smiles-col "smiles_neutral" \\
        --id-col "row_uid" \\
        --target "logS" \\
        --weight-col ${weight_col} \\
        --epochs ${epochs} \\
        --batch-size ${batch_size} \\
        --tune-trials ${tune_trials} \\
        --tune-pruner ${tune_pruner} \\
        --asha-rungs ${asha_rungs} \\
        --save-dir ./oof_gnn
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
