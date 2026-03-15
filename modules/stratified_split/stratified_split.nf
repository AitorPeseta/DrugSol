#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Stratified Split
========================================================================================
    Description:
    Performs scaffold-aware stratified train/test splitting to prevent data leakage
    and ensure balanced evaluation. The split respects molecular scaffold clusters
    (structurally similar molecules stay together) while maintaining target distribution.
    
    Splitting Strategy:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  1. Group molecules by scaffold cluster (from Butina clustering)               │
    │  2. Assign stratification label based on logS bin + temperature bin            │
    │  3. Collapse to group level (majority label per cluster)                       │
    │  4. Split groups to achieve target ratio while preserving stratification       │
    │  5. Map group assignments back to individual molecules                         │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Why Scaffold Splitting:
    - Prevents data leakage from structurally similar compounds
    - More realistic evaluation: models must generalize to new scaffolds
    - Standard practice in molecular property prediction
    
    Input:
    - Balanced Parquet file with scaffold cluster assignments
    
    Output:
    - train.parquet: Training set (~80% of data)
    - test.parquet: Test set (~20% of data)
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: STRATIFIED_SPLIT
========================================================================================
*/

process stratified_split {
    tag "Split Iter #${meta_id}"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/split", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(balanced_file)  // Balanced dataset with clusters
        val  outdir_base                          // Output directory (for logging)
        path script_py                            // Python script: stratified_split.py
        val seed_fixed                            // Random seed for splitting

    output:
        tuple val(meta_id), path("train.parquet"), path("test.parquet"), emit: splits

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
    
        // Temperature binning step (°C)
        def temp_step = params.split_temp_step ?: 5
        
        // Fraction of data for test set
        def test_size = params.split_test_size ?: 0.2
        
    """
    python ${script_py} \\
        --input "${balanced_file}" \\
        --group-col "cluster_ecfp4_0p7" \\
        --temp-col "temp_C" \\
        --temp-step ${temp_step} \\
        --test-size ${test_size} \\
        --seed ${seed_fixed} \\
        --outdir .
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
