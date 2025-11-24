nextflow.enable.dsl = 2

process detect_outliers {
    tag "Detect Outliers (Robust Z-Score)"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file      // Filtered by temperature parquet file
        val  outdir_val
        path script_py        // Python script

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