nextflow.enable.dsl = 2

process fetch_bigsoldb {
    tag "BigSolDB-${record_id}"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/ingest", mode: 'copy', overwrite: true

    input:
        val  outdir_val  
        val  record_id    // Zenodo record ID
        path script_py    // Python script

    output:
        path "bigsoldb.csv", emit: out

    script:
    """
    python ${script_py} \\
        --record ${record_id} \\
        --kind main \\
        --out bigsoldb.csv \\
        --normalize
    """
}