#!/usr/bin/env nextflow
/*
========================================================================================
    Module: detect_outliers
========================================================================================
    Detects statistical outliers in solubility data using temperature-conditioned
    robust z-scores.
    
    Why Temperature-Conditioned?
    - Solubility varies significantly with temperature (Van't Hoff relationship)
    - A "normal" logS at 25°C might be an outlier at 50°C
    - Binning by temperature ensures fair comparison within similar conditions
    
    Method: Robust Z-Score (Median Absolute Deviation)
    - More resistant to outliers than standard z-score (mean/std)
    - Z = (x - median) / (MAD × 1.4826)
    - Threshold: |Z| > 3.0 flagged as outlier
    
    Algorithm:
    1. Bin data by temperature (equal-width bins)
    2. Calculate robust z-score within each bin
    3. Flag points where |z| > threshold
    4. Output original data + is_outlier column
        
    Configurable parameters (hardcoded defaults, could be exposed):
    - binning: width (equal-width bins)
    - bins: 10
    - z-method: robust (median/MAD)
    - z-thresh: 3.0
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
----------------------------------------------------------------------------------------
    Process: detect_outliers
----------------------------------------------------------------------------------------
    Flags statistical outliers in logS using temperature-binned robust z-scores.
    
    Input:
    - source_file: Temperature-filtered Parquet file
    - outdir_val:  Output directory path
    - script_py:   Python outlier detection script
    
    Output:
    - detect_outliers.parquet: Original data + is_outlier column (0/1)
----------------------------------------------------------------------------------------
*/

process detect_outliers {
    tag "Detect Outliers (Robust Z-Score)"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file    // Temperature-filtered Parquet
        val  outdir_val     // Output directory path
        path script_py      // Python detection script

    output:
        path "detect_outliers.parquet", emit: out

    script:
    """
    python ${script_py} \\
        --input ${source_file} \\
        --out detect_outliers.parquet \\
        --binning width \\
        --bins 10 \\
        --z-method robust \\
        --z-thresh 3.0 \\
        --export-csv
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
