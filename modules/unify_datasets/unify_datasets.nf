#!/usr/bin/env nextflow
/*
========================================================================================
    Module: unify_datasets
========================================================================================
    Combines multiple solubility datasets into a unified standardized schema.
    
    This module handles the challenge of integrating data from diverse sources
    (BigSolDB, ChEMBL, Reaxys, Challenge datasets) that use different column
    names, units, and formats.
    
    Schema Standardization:
    ┌─────────────────┬─────────────────────────────────────────────────────────┐
    │ Output Column   │ Description                                             │
    ├─────────────────┼─────────────────────────────────────────────────────────┤
    │ smiles_original │ Original SMILES string from source                      │
    │ smiles_neutral  │ Neutralized SMILES                                      │
    │ solvent         │ Solvent identifier (typically 'water')                  │
    │ temp_C          │ Temperature in Celsius (auto-converted from Kelvin)     │
    │ logS            │ Log10 of molar solubility (mol/L)                       │
    │ is_outlier      │ Outlier flag (populated in curation step)               │
    │ source          │ Data origin identifier                                  │
    └─────────────────┴─────────────────────────────────────────────────────────┘
    
    Features:
    - Automatic column name mapping from common variations
    - Temperature unit detection and conversion (K → C)
    - Robust file reading (CSV/Parquet with encoding fallbacks)
    - Source tracking for data provenance
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
----------------------------------------------------------------------------------------
    Process: unify_datasets
----------------------------------------------------------------------------------------
    Merges multiple data sources into a single standardized Parquet file.
    
    Input:
    - source_files: Collection of CSV/Parquet files to merge
    - outdir_val:   Output directory path
    - script_py:    Python unification script
    
    Output:
    - unified.parquet: Combined dataset with standardized schema
----------------------------------------------------------------------------------------
*/

process unify_datasets {
    tag "Unify Datasets"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/ingest", mode: 'copy', overwrite: true

    input:
        path source_files   // Collection of input files (CSV/Parquet)
        val  outdir_val     // Output directory path
        path script_py      // Python unification script

    output:
        path "unified.parquet", emit: parquet

    script:
    """
    python ${script_py} \\
        --sources ${source_files} \\
        --output unified
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
