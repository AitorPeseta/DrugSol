#!/usr/bin/env nextflow
/*
========================================================================================
    Module: filter_water
========================================================================================
    Filters dataset to retain only aqueous solubility measurements.
    
    Aqueous solubility is the primary focus of drug solubility prediction since:
    - Most biological systems are aqueous
    - Drug absorption typically occurs in aqueous media
    - Regulatory guidelines focus on aqueous solubility
    
    This filter removes measurements in organic solvents (ethanol, DMSO, etc.)
    and mixed solvent systems, ensuring model training on consistent data.
    
    Filter Logic:
    - Case-insensitive match for 'water' in solvent column
    - Whitespace-tolerant comparison
    - Logs statistics for QC purposes
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
----------------------------------------------------------------------------------------
    Process: filter_water
----------------------------------------------------------------------------------------
    Filters unified dataset to keep only water-based solubility measurements.
    
    Input:
    - source_file: Unified Parquet file from ingestion
    - outdir_val:  Output directory path
    - script_py:   Python filtering script
    
    Output:
    - filter_water.parquet: Dataset with only aqueous solubility entries
----------------------------------------------------------------------------------------
*/

process filter_water {
    tag "Filter Water"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file    // Unified Parquet from ingestion step
        val  outdir_val     // Output directory path
        path script_py      // Python filtering script

    output:
        path "filter_water.parquet", emit: out

    script:
    """
    python ${script_py} \\
        --input ${source_file} \\
        --output filter_water.parquet
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
