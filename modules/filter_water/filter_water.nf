nextflow.enable.dsl = 2

process filter_water {
    tag "Filter Water"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file   // Unified parquet file
        val  outdir_val    
        path script_py     // Python script

    output:
        path "filter_water.parquet", emit: out

    script:
    """
    python ${script_py} \\
        --input ${source_file} \\
        --output filter_water.parquet
    """
}