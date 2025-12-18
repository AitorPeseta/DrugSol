nextflow.enable.dsl = 2

process fetch_chembl {
    tag "CHEMBL"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/ingest", mode: 'copy', overwrite: true

    input:
        path raw_csv
        path script_py    // Python script

    output:
        path "chembl.csv", emit: out

    script:
    """
    python ${script_py} --input ${raw_csv}

    """
}