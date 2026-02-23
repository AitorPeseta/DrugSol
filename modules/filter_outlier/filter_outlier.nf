nextflow.enable.dsl = 2

process filter_outlier {
    tag "Filter Outliers"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file    // Parquet file with outlier flags
        val  outdir_val
        path script_py      // Python script

    output:
        path "filter_outlier.parquet", emit: out

    script:
    """
    # Filter rows where is_outlier == 1
    python ${script_py} \\
        --input ${source_file} \\
        --out filter_outlier.parquet \\
        --save-csv
    """
}