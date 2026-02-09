#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Align Feature Columns
========================================================================================
    Description:
    Ensures the test dataset has exactly the same columns (and order) as the
    training dataset. Critical for model inference where feature alignment
    is required.
    
    Alignment Operations:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Operation         │ Action                                                    │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Extra columns     │ Removed from test (not seen during training)             │
    │  Missing columns   │ Added to test with appropriate fill values               │
    │  Column order      │ Reordered to match training exactly                      │
    │  One-hot columns   │ Missing solvent indicators filled with 0                 │
    │  Numeric columns   │ Missing values filled with NaN (handled by imputer)      │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Why This Matters:
    - ML models expect identical feature space at training and inference
    - Column order affects some model implementations
    - Missing one-hot columns should be 0, not NaN
    
    Input:
    - Reference training file (defines expected columns)
    - Test file to align
    
    Output:
    - Aligned test file with identical columns to training
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: ALIGN_FEATURE_COLUMNS
========================================================================================
*/

process align_feature_columns {
    tag "Align Test to Train"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/aligned", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(test_file)  // Reference train and target test
        val  outdir_val                                         // Output directory (for logging)
        path script_py                                          // Python script: align_feature_columns.py

    output:
        tuple val(meta_id), path("features_test_aligned.parquet"), emit: out

    script:
        
    """
    python ${script_py} \\
        --train "${train_file}" \\
        --test "${test_file}" \\
        --out "features_test_aligned.parquet" \\
        --onehot-prefix "solv_"
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
