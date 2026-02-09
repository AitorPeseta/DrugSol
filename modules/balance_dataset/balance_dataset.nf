#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Balance Dataset
========================================================================================
    Description:
    Performs intelligent dataset balancing to reduce temperature bias while
    preserving valuable physiological temperature data. Room temperature (25°C)
    measurements are abundant in literature but less relevant for drug solubility
    at body temperature.
    
    Balancing Strategy:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Temperature Range    │ Action                                                  │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Ambient (20-25°C)    │ Undersample: Limit to N samples per logS bin           │
    │  Normal (25-35°C)     │ Keep all: Transitional temperature range               │
    │  Physiological (35-40°C) │ Keep all: Most relevant for drug applications       │
    │  Hot (40-50°C)        │ Keep all: Valuable high-temperature data               │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    The balancing is performed within 2D bins (logS × temperature) to maintain
    the solubility distribution while reducing ambient temperature dominance.
    
    Input:
    - Curated Parquet file with logS and temperature columns
    
    Output:
    - Balanced Parquet file with reduced ambient temperature samples
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: BALANCE_DATASET
========================================================================================
*/

process balance_dataset {
    tag "Balance Iter #${iter_id}"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"

    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:
        tuple val(iter_id), path(input_parquet)  // Curated dataset
        path script_py                            // Python script: balance_dataset.py
        val seed_base                             // Base random seed

    output:
        tuple val(iter_id), path("balanced_${iter_id}.parquet"), emit: balanced_data

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        // Maximum samples per bin for ambient temperature data
        def limit_ambient = params.balance_limit_ambient ?: 30
        
        // Size of logS bins for stratification (smaller = finer control)
        def bin_size = params.balance_bin_size ?: 0.2
        
    """
    CURRENT_SEED=\$(( ${seed_base} + ${iter_id} ))

    python ${script_py} \\
        --input "${input_parquet}" \\
        --output "balanced_${iter_id}.parquet" \\
        --limit ${limit_ambient} \\
        --bin-size ${bin_size} \\
        --seed \$CURRENT_SEED
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
