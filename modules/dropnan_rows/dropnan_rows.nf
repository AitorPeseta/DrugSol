#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Drop NaN Rows
========================================================================================
    Description:
    Final data cleaning step that removes rows with missing critical values.
    Uses intelligent filtering that distinguishes between critical columns
    (IDs, targets) and feature columns (where NaN is acceptable for imputation).
    
    Cleaning Strategy:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Mode    │ Critical Columns Checked    │ Feature Columns                       │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  TRAIN   │ IDs (row_uid, smiles) + logS │ NaN allowed (imputed later)          │
    │  TEST    │ IDs (row_uid, smiles) only   │ NaN allowed (imputed later)          │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Important: This module does NOT drop rows just because a Mordred descriptor
    failed to compute. Only rows missing critical identification or target
    information are removed.
    
    Input:
    - Parquet file with features and metadata
    
    Output:
    - Cleaned Parquet file with valid rows only
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: DROPNAN_ROWS
========================================================================================
*/

process dropnan_rows {
    tag "Clean Rows (${mode})"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/final", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(source_file)  // Input parquet to clean
        val  outdir_val                         // Output directory (for logging)
        path script_py                          // Python script: dropnan_rows.py
        val  name_out                           // Output filename (e.g., "final_train_gbm")
        val  mode                               // "train" or "test"
        val  subset                             // Subset identifier

    output:
        tuple val(meta_id), path("${name_out}.parquet"), emit: out

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        
        // Subset mode for cleaning
        def subset_mode = params.dropnan_subset_mode ?: 'features_only'
        
    """
    python ${script_py} \\
        --input "${source_file}" \\
        --output "${name_out}.parquet" \\
        --mode "${mode}" \\
        --subset "${subset_mode}"
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
