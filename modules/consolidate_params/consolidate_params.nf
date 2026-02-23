#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Consolidate Parameters
========================================================================================
    Description:
    Aggregates hyperparameters from multiple fold-specific JSON files into a single
    consolidated parameter set for final model training.
    
    Aggregation Strategy:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Parameter Type   │ Aggregation Method                                         │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Numeric (float)  │ Mean of values across folds                               │
    │  Numeric (int)    │ Rounded mean of values across folds                       │
    │  Categorical      │ Mode (most frequent value)                                 │
    │  Boolean          │ Mode (most frequent value)                                 │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    This ensures that the final model uses representative hyperparameters that
    reflect the optimal settings discovered across all cross-validation folds.
    
    Input:
    - Multiple JSON files with fold-specific hyperparameters
    
    Output:
    - Single consolidated JSON file
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: CONSOLIDATE_PARAMS
========================================================================================
*/

process consolidate_params {
    tag "Consolidating HPs"
    label 'process_low'
    
    conda "${params.conda_env_data}"

    input:
        path json_files, stageAs: 'input_?'  // Accepts files or directories
        path script_consolidate               // Python script

    output:
        path "best_params_consolidated.json", emit: BEST_PARAMS

    script:
    """
    python3 ${script_consolidate} ${json_files} --output best_params_consolidated.json
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
