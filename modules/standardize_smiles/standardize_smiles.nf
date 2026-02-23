#!/usr/bin/env nextflow
/*
========================================================================================
    Module: standardize_smiles
========================================================================================
    Standardizes molecular structures and generates unique identifiers.
    
    This is a critical step for ensuring consistent molecular representations
    across the dataset, which is essential for:
    - Accurate duplicate detection
    - Consistent fingerprint generation
    - Reproducible model training
    
    Standardization Pipeline (RDKit MolStandardize):
    1. Parse SMILES string
    2. Remove salts and counterions (single fragment only)
    3. Normalize functional groups (nitro, sulfo, etc.)
    4. Reionize to consistent protonation state
    5. Neutralize charges where appropriate
    6. Canonicalize tautomers (optional)
    7. Generate canonical SMILES and InChIKey
    
    Deduplication Logic:
    - Groups by InChIKey + solvent + temperature
    - If logS values within threshold: average them
    - If logS values conflict (diff > threshold): remove all (unreliable)
    
    Configurable via params:
    - params.dedup_threshold: Maximum logS difference for consensus (default: 0.7)
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
----------------------------------------------------------------------------------------
    Process: standardize_smiles
----------------------------------------------------------------------------------------
    Standardizes SMILES, generates InChIKeys, and deduplicates entries.
    
    Input:
    - source_file: Input Parquet (from outlier filter or temperature filter)
    - outdir_val:  Output directory path
    - script_py:   Python standardization script
    
    Output:
    - standardize.parquet: Standardized and deduplicated dataset with:
                           smiles_neutral, InChIKey, row_uid columns added
----------------------------------------------------------------------------------------
*/

process standardize_smiles {
    tag "Standardize Structure"
    label 'cpu_medium'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file    // Input Parquet file
        val  outdir_val     // Output directory path
        path script_py      // Python standardization script

    output:
        path "standardize.parquet", emit: out

    script:
    // Deduplication threshold: max logS difference to consider values consistent
    def dedup_thresh = params.dedup_threshold ?: 0.7
    """
    python ${script_py} \\
        --in ${source_file} \\
        --out standardize.parquet \\
        --overwrite-inchikey \\
        --dedup \\
        --dedup-thresh ${dedup_thresh} \\
        --export-csv \\
        --engine auto
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
