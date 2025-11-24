nextflow.enable.dsl = 2

process unify_datasets {
    tag "Unify Datasets"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/ingest", mode: 'copy', overwrite: true

    input:
        path source_files  // Can be one file or a list of files
        val  outdir_val    
        path script_py     // The python script

    output:
        path "unified.parquet", emit: parquet

    script:
    """
    python ${script_py} \\
        --sources ${source_files} \\
        --export-csv
    """
}