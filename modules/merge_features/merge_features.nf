#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Merge Features
========================================================================================
    Description:
    Combines multiple feature sets into a single unified DataFrame for model training.
    Primary use case: Merging Mordred molecular descriptors with ChemBERTa embeddings.
    
    Features:
    - Inner join on specified ID column (only keeps matching rows)
    - Handles column name conflicts with configurable suffixes
    - Validates row counts before and after merge
    - Supports merging 2+ feature sets
    
    Input:
    - Primary feature file (e.g., Mordred descriptors)
    - Secondary feature file (e.g., ChemBERTa embeddings)
    - Both must share a common ID column
    
    Output:
    - Single Parquet file with all features combined
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: MERGE_FEATURES
========================================================================================
*/

process merge_features {
    tag "Merge Features #${iter_id} (${split_name})"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:
        tuple val(iter_id), path(primary_file), path(secondary_file)  // Two feature files to merge
        val  outdir_val                                                // Output directory (for logging)
        path script_py                                                 // Python script: merge_features.py
        val  split_name                                                // "train" or "test" (for output naming)

    output:
        tuple val(iter_id), path("${split_name}_merged_features.parquet"), emit: out

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
    
        // How to handle rows that don't match
        // Options: "inner" (only matching), "left" (keep all primary), "outer" (keep all)
        def merge_how = params.merge_strategy ?: 'inner'
        
        // Whether to validate row counts match after merge
        def validate = params.merge_validate ? '--validate' : ''
        
    """
    #!/usr/bin/env bash
    set -euo pipefail
   
    python ${script_py} \\
        --primary "${primary_file}" \\
        --secondary "${secondary_file}" \\
        --output "${split_name}_merged_features.parquet" \\
        --merge-col "row_uid" \\
        --how "${merge_how}" \\
        --suffix-primary " " \\
        --suffix-secondary "_bert" \\
        ${validate}
    """
}

/*
========================================================================================
    PROCESS: MERGE_FEATURES_MULTI
========================================================================================
    Alternative process for merging more than 2 feature files.
    Accepts a list of files and merges them sequentially.
*/

process merge_features_multi {
    tag "Merge ${file_list.size()} Feature Sets #${iter_id} (${split_name})"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:
        tuple val(iter_id), path(file_list)  // List of feature files to merge
        val  outdir_val                       // Output directory
        path script_py                        // Python script
        val  split_name                       // "train" or "test"

    output:
        tuple val(iter_id), path("${split_name}_merged_features.parquet"), emit: out

    script:
        def merge_col = params.merge_id_col ?: 'row_uid'
        def merge_how = params.merge_strategy ?: 'inner'
        
        // Convert file list to space-separated string
        def files_str = file_list.join(' ')
        
    """
    #!/usr/bin/env bash
    set -euo pipefail
    
    python ${script_py} \\
        --files ${files_str} \\
        --output "${split_name}_merged_features.parquet" \\
        --merge-col "${merge_col}" \\
        --how "${merge_how}"
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
