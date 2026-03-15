#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Train Full Chemprop
========================================================================================
    Description:
    Trains the final Chemprop D-MPNN model on the complete dataset using
    hyperparameters optimized during OOF cross-validation.
    
    Training Strategy:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Pure Graph Approach                                                            │
    │  - Uses only SMILES structure (no external descriptors)                         │
    │  - Consistent with OOF training strategy                                        │
    │  - Sample weighting via data_weights_path                                       │
    │  - Minimal validation set (5 samples) for Chemprop requirements                 │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    The model learns molecular representations directly from the graph structure,
    avoiding potential noise from pre-computed descriptors.
    
    Input:
    - Full training data with SMILES
    - Best hyperparameters from OOF tuning
    
    Output:
    - Trained Chemprop model directory
    - Model manifest for inference
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: TRAIN_FULL_CHEMPROP
========================================================================================
*/

process train_full_chemprop {
    tag "Train Full Chemprop"
    label 'process_gpu'
    
    conda "${params.conda_env_train}"
    
    publishDir "${params.outdir}/training/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(best_params_json)  // Data and hyperparameters
        val  outdir_val                                                // Output directory
        path script_py                                                 // Python script

    output:
        tuple val(meta_id), path("models_GNN"),                       emit: CHEMPROP_DIR
        tuple val(meta_id), path("models_GNN/chemprop_manifest.json"), emit: MANIFEST

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        def weight_col = params.full_chemprop_weight_col ?: 'weight'
        def epochs = params.full_chemprop_epochs ?: 50
        def batch_size = params.full_chemprop_batch_size ?: 50
        def use_gpu = params.full_chemprop_use_gpu != false ? '--gpu' : ''
        
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
        --smiles-col "smiles_neutral" \\
        --target "logS" \\
        --best-params "${best_params_json}" \\
        ${use_gpu} \\
        --epochs ${epochs} \\
        --batch-size ${batch_size} \\
        --weight-col ${weight_col} \\
        --save-dir models_GNN
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
