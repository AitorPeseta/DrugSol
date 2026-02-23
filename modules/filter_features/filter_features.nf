#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Filter Features
========================================================================================
    Description:
    Advanced feature selection pipeline to reduce dimensionality and remove
    redundant features before model training. Operates on Mordred descriptors
    to select the most informative non-redundant feature subset.
    
    Feature Selection Pipeline:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Step 1: Remove Constant/NZV Columns                                            │
    │          - Drop features with zero variance                                     │
    │          - Drop features where >99% of values are identical                     │
    │                                                                                 │
    │  Step 2: Correlation Clustering                                                 │
    │          - Build correlation graph (edges where |r| > threshold)                │
    │          - Find connected components (clusters of correlated features)          │
    │                                                                                 │
    │  Step 3: Importance-Based Selection                                             │
    │          - Train quick LightGBM/XGBoost/CatBoost model                          │
    │          - Select feature with highest gain importance per cluster              │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    This approach typically reduces ~1800 Mordred descriptors to ~400-700 features
    while preserving predictive information.
    
    Input:
    - Parquet file with Mordred descriptors (prefixed columns)
    
    Output:
    - Parquet file with filtered feature subset
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: FILTER_FEATURES
========================================================================================
*/

process filter_features {
    tag "Filter Features (${dataset_name})"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/filtered", mode: 'copy', overwrite: true
    
    input:
        tuple val(meta_id), path(input_file)  // Input parquet with Mordred features
        val  outdir_val                        // Output directory (for logging)
        path script_py                         // Python script: filter_features.py
        val  dataset_name                      // "train" or "test" (for output naming)

    output:
        tuple val(meta_id), path("${dataset_name}_features_mordred_filtered.parquet"), emit: out
    
    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        
        // Correlation threshold for clustering (features with |r| > threshold are grouped)
        def corr_thresh = params.filter_corr_threshold ?: 0.99
        
        // Algorithm for importance calculation
        def algo = params.filter_importance_algo ?: 'lgbm'
        
    """
    python ${script_py} \\
        --input "${input_file}" \\
        --output "${dataset_name}_features_mordred_filtered.parquet" \\
        --target "logS" \\
        --mordred-prefix "mordred__" \\
        --corr-thresh ${corr_thresh} \\
        --algo ${algo}
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
