#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Make Fingerprints
========================================================================================
    Description:
    Generates molecular fingerprints and performs scaffold-based clustering for
    stratified train/test splitting. Uses Morgan fingerprints (ECFP4) with
    Butina clustering based on Tanimoto similarity.
    
    Pipeline Steps:
    1. Parse SMILES to RDKit Mol objects
    2. Generate Morgan fingerprints (radius=2, equivalent to ECFP4)
    3. Expand fingerprint bits to individual columns (for GBM models)
    4. Calculate pairwise Tanimoto similarity matrix
    5. Perform Butina clustering with specified similarity cutoff
    6. Assign cluster IDs for scaffold-based splitting
    
    Fingerprint Specification:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Parameter       │ Value    │ Description                                       │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Type            │ Morgan   │ Circular fingerprint based on atom environments   │
    │  Radius          │ 2        │ Equivalent to ECFP4 (diameter 4 bonds)            │
    │  Bits            │ 2048     │ Fingerprint vector length (configurable)          │
    │  Clustering      │ Butina   │ Taylor-Butina clustering algorithm                │
    │  Cutoff          │ 0.7      │ Tanimoto similarity threshold (configurable)      │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Input:
    - Parquet file with standardized SMILES column
    
    Output:
    - Parquet file with fingerprint bit columns and cluster assignments
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: MAKE_FINGERPRINTS
========================================================================================
*/

process make_fingerprints {
    tag "ECFP4 & Butina Clustering #${meta_id}"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/fingerprints", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(input_file)  // Input parquet with standardized SMILES
        val  outdir_val                       // Output directory (for logging)
        path script_py                        // Python script: make_fingerprints.py
        val  name_prefix                      // Output file prefix (e.g., "cluster_ecfp4_0p7")

    output:
        tuple val(meta_id), path("${name_prefix}_fingerprint.parquet"), emit: out

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
       
        // Fingerprint parameters
        def n_bits = params.fingerprint_n_bits ?: 2048
        def radius = params.fingerprint_radius ?: 2  // radius=2 → ECFP4
        
        // Clustering parameters
        // Cutoff is Tanimoto similarity threshold (0.7 = 70% similar)
        def cluster_cutoff = params.fingerprint_cluster_cutoff ?: 0.7
        
        
    """
    #!/usr/bin/env bash
    set -euo pipefail
    
    python ${script_py} \\
        --input "${input_file}" \\
        --out-parquet "${name_prefix}_fingerprint.parquet" \\
        --smiles-col "smiles_neutral" \\
        --n-bits ${n_bits} \\
        --radius ${radius} \\
        --cluster-cutoff ${cluster_cutoff}
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
