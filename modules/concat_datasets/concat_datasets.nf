#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Concat Datasets
========================================================================================
    Description:
    Concatenates train and test datasets into a single full dataset. Used for
    final model training on all available data after validation is complete.
    
    Operations:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  1. Load train and test parquet files                                           │
    │  2. Concatenate vertically (append rows)                                        │
    │  3. Remove 'fold' column (no longer meaningful)                                 │
    │  4. Remove duplicates by row_uid (safety measure)                               │
    │  5. Save combined dataset                                                       │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Use Case:
    After cross-validation confirms model quality, we retrain on the full dataset
    (train + test) to maximize the data available for the production model.
    
    Input:
    - Training dataset parquet
    - Test dataset parquet
    
    Output:
    - Combined full dataset parquet
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: CONCAT_DATASETS
========================================================================================
*/

process concat_datasets {
    tag "Merge Train+Test"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"

    input:
        path train_file   // Training data parquet
        path test_file    // Test data parquet
        path script_py    // Python script

    output:
        path "full_dataset.parquet", emit: OUT

    script:
    """
    python3 ${script_py} \\
        --train ${train_file} \\
        --test ${test_file} \\
        --out full_dataset.parquet
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
