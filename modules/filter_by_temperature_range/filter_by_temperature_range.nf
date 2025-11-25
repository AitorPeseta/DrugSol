nextflow.enable.dsl = 2

process filter_by_temperature_range {
    tag "Filter Temp [${min_val}, ${max_val}]"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file      // Parquet file to filter
        val  outdir_val
        path script_py        // Python script
        val  min_val          // Minimum temperature
        val  max_val          // Maximum temperature

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