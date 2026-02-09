#!/usr/bin/env nextflow
/*
========================================================================================
    Module: fetch_chembl
========================================================================================
    Processes pre-downloaded ChEMBL solubility data for pipeline integration.
    
    ChEMBL is a manually curated database of bioactive molecules with drug-like
    properties. This module extracts and standardizes solubility measurements
    from a local ChEMBL export file.
    
    Processing Steps:
    1. Parse ChEMBL CSV export (handles various delimiters)
    2. Convert diverse solubility units to standardized logS (mol/L)
    3. Filter for poorly soluble compounds (logS < -5)
    4. Extract temperature from assay descriptions
    5. Format output for pipeline compatibility
    
    Supported Input Units:
    - Molar: M, mM, µM, nM
    - Mass/Volume: mg/mL, µg/mL, ng/mL, g/L
    - Pre-computed: logS values
    
    Note: Requires pre-downloaded chembl_raw.csv in resources/ directory
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
----------------------------------------------------------------------------------------
    Process: fetch_chembl
----------------------------------------------------------------------------------------
    Processes ChEMBL solubility data and converts to standardized format.
    
    Input:
    - raw_csv:   Pre-downloaded ChEMBL CSV file (from resources/)
    - script_py: Python processing script
    
    Output:
    - chembl.csv: Standardized solubility data with columns:
                  [smiles_original, logS, solvent, temp_C, source, is_temp_assumed]
----------------------------------------------------------------------------------------
*/

process fetch_chembl {
    tag "ChEMBL"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/ingest", mode: 'copy', overwrite: true

    input:
        path raw_csv      // Pre-downloaded ChEMBL CSV file
        path script_py    // Python processing script

    output:
        path "chembl.csv", emit: out

    script:
    """
    python ${script_py} --input ${raw_csv}
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
