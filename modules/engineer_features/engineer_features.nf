#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Engineer Features
========================================================================================
    Description:
    Calculates physicochemical and engineered features for solubility prediction.
    These features complement molecular descriptors with domain-specific knowledge.
    
    Features Computed:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Feature              │ Description                                            │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  weight               │ Sample weight (Gaussian @ 37°C + logS rarity)          │
    │  n_ionizable          │ Count of ionizable groups (acid + base)                │
    │  n_acid               │ Number of acidic ionization sites                      │
    │  n_base               │ Number of basic ionization sites                       │
    │  n_phenol             │ Count of phenolic hydroxyl groups                      │
    │  has_phenol           │ Binary indicator for phenol presence                   │
    │  pka_acid_min/max     │ Predicted acidic pKa range (via MolGpKa API)           │
    │  pka_base_min/max     │ Predicted basic pKa range (via MolGpKa API)            │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Sample Weighting Strategy:
    - Temperature: Gaussian centered at 37°C (physiological) with σ=5°C
    - Solubility: Log-scaled inverse frequency weighting for rare logS bins
    - Combined via quadratic scaling: weight = w_solubility × (w_temp)²
    
    Input:
    - Parquet file with SMILES and temperature columns
    
    Output:
    - Parquet file with additional engineered feature columns
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: ENGINEER_FEATURES
========================================================================================
*/

process engineer_features {
    tag "Physics & Chemistry #${meta_id}"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/engineer", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(input_parquet)  // Input parquet with standardized SMILES
        val  outdir_val                          // Output directory (for logging)
        path script_py                           // Python script: engineer_features.py

    output:
        tuple val(meta_id), path("engineered_features.parquet"), emit: out

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        
        // SMARTS patterns file for ionization detection
        def smarts_file = params.smarts_patterns ?: "${baseDir}/resources/smarts_pattern_ionized.txt"
        
        // pKa prediction API configuration
        // External API: MolGpKa (http://xundrug.cn:5001)
        def pka_api_url = params.pka_api_url ?: 'http://xundrug.cn:5001/modules/upload0/'
        def pka_token   = params.pka_api_token ?: 'O05DriqqQLlry9kmpCwms2IJLC0MuLQ7'
        
    """
    #!/usr/bin/env bash
    set -euo pipefail
   
    python ${script_py} \\
        --in "${input_parquet}" \\
        --out "engineered_features.parquet" \\
        --smiles-col "smiles_neutral" \\
        --temp-col "temp_C" \\
        --smarts "${smarts_file}" \\
        --pka-api-url "${pka_api_url}" \\
        --pka-token "${pka_token}" \\
        --nproc ${task.cpus}
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
