#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Make Folds
========================================================================================
    Description:
    Generates cross-validation fold assignments for model training. Uses grouped
    stratified K-fold splitting to ensure:
    
    1. Scaffold Grouping: Molecules in the same scaffold cluster stay together
       (prevents data leakage from structurally similar compounds)
    
    2. Target Stratification: Each fold has similar distribution of logS values
       (ensures balanced representation of solubility ranges)
    
    Splitting Strategy:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Step │ Action                                                                  │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  1    │ Group molecules by scaffold cluster (from Butina clustering)           │
    │  2    │ Assign stratification label based on logS bin                          │
    │  3    │ Collapse to group level (majority label per cluster)                   │
    │  4    │ Apply StratifiedKFold on groups                                        │
    │  5    │ Map fold assignments back to individual molecules                      │
    │  6    │ Handle edge cases (rare classes, orphan rows)                          │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Input:
    - Training data Parquet with cluster assignments and target column
    
    Output:
    - Parquet file with row_uid, fold, and cluster_id columns
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: MAKE_FOLDS
========================================================================================
*/

process make_folds {
    tag "Generate CV Folds"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/training/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file)  // Training data with cluster assignments
        val  outdir_val                        // Output directory (for logging)
        path script_py                         // Python script: make_folds.py

    output:
        tuple val(meta_id), path("folds.parquet"), emit: out

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
       
        // Target column for stratification
        def target_col = params.folds_target_col ?: 'logS'
        
        // Number of CV folds
        def n_splits = params.n_cv_folds ?: 5
        
        // Random seed for reproducibility
        def seed = params.random_seed ?: 42
        
        // Number of bins for target stratification
        def n_bins = params.folds_n_bins ?: 5
        
    """
    #!/usr/bin/env bash
    set -euo pipefail
    
    python ${script_py} \\
        --input "${train_file}" \\
        --out "folds.parquet" \\
        --id-col "row_uid" \\
        --group-col "cluster_ecfp4_0p7" \\
        --target "${target_col}" \\
        --n-splits ${n_splits} \\
        --bins ${n_bins} \\
        --seed ${seed} \\
        --temp-col "temp_C" \\
        --strat-mode "both"
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
