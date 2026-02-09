#!/usr/bin/env nextflow
/*
========================================================================================
    Module: filter_by_temperature_range
========================================================================================
    Filters solubility data to a specified temperature range.
    
    Temperature filtering is critical for solubility modeling because:
    - Solubility is strongly temperature-dependent (Van't Hoff equation)
    - Most drug applications target physiological temperatures
    - Extreme temperatures may introduce measurement artifacts
    - Consistent temperature range reduces model complexity
    
    Default Range: 24-50°C
    - Lower bound (24°C): Slightly below room temperature
    - Upper bound (50°C): Above body temperature, captures fever conditions
    
    This range covers:
    - Room temperature storage conditions (~25°C)
    - Physiological temperature (~37°C)
    - Elevated temperatures for accelerated stability studies
    
    Configurable via params:
    - params.temp_min_celsius (default: 24)
    - params.temp_max_celsius (default: 50)
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
----------------------------------------------------------------------------------------
    Process: filter_by_temperature_range
----------------------------------------------------------------------------------------
    Filters dataset to retain only measurements within specified temperature bounds.
    
    Input:
    - source_file: Input Parquet file (from water filter step)
    - outdir_val:  Output directory path
    - script_py:   Python filtering script
    - min_val:     Minimum temperature in Celsius (inclusive)
    - max_val:     Maximum temperature in Celsius (inclusive)
    
    Output:
    - filter_temp.parquet: Temperature-filtered dataset
----------------------------------------------------------------------------------------
*/

process filter_by_temperature_range {
    tag "Filter Temp [${min_val}°C, ${max_val}°C]"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file    // Input Parquet file
        val  outdir_val     // Output directory path
        path script_py      // Python filtering script
        val  min_val        // Minimum temperature (Celsius)
        val  max_val        // Maximum temperature (Celsius)

    output:
        path "filter_temp.parquet", emit: out

    script:
    """
    python ${script_py} \\
        --input ${source_file} \\
        --out filter_temp.parquet \\
        --temp-col temp_C \\
        --min ${min_val} \\
        --max ${max_val}
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
