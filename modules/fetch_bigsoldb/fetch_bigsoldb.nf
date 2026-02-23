#!/usr/bin/env nextflow
/*
========================================================================================
    Module: fetch_bigsoldb
========================================================================================
    Downloads the BigSolDB dataset from Zenodo repository.
    
    BigSolDB is the largest open-source aqueous solubility database containing
    experimental solubility measurements for thousands of compounds.
    
    Features:
    - Automatic file detection from Zenodo record
    - MD5 checksum verification
    - GZIP decompression support
    - CSV normalization (UTF-8, Unix line endings)
    
    Reference:
    - BigSolDB: https://zenodo.org/records/15094979
    - Provides standardized SMILES, logS values, temperature, and solvent info
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
----------------------------------------------------------------------------------------
    Process: fetch_bigsoldb
----------------------------------------------------------------------------------------
    Downloads and normalizes BigSolDB solubility data from Zenodo.
    
    Input:
    - outdir_val:  Output directory path (for logging purposes)
    - record_id:   Zenodo record ID (e.g., '15094979' for latest version)
    - script_py:   Python download script path
    
    Output:
    - bigsoldb.csv: Normalized CSV with solubility measurements
    
    Note: The Zenodo record ID can be changed via params.bigsoldb_zenodo_id
    to use different versions of the database.
----------------------------------------------------------------------------------------
*/

process fetch_bigsoldb {
    tag "BigSolDB-${record_id}"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/ingest", mode: 'copy', overwrite: true

    input:
        val  outdir_val     // Output directory path
        val  record_id      // Zenodo record ID (e.g., '15094979')
        path script_py      // Python download script

    output:
        path "bigsoldb.csv", emit: out

    script:
    """
    python ${script_py} \\
        --record ${record_id} \\
        --kind main \\
        --out bigsoldb.csv \\
        --normalize
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
