nextflow.enable.dsl = 2

process standardize_smiles {
    tag "Standardize Structure"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/curate", mode: 'copy', overwrite: true

    input:
        path source_file   // Input parquet
        val  outdir_val    
        path script_py     // Python script

    output:
        path "standardize.parquet", emit: out

    script:
    """
    # Run standardization with RDKit
    python ${script_py} \\
        --in ${source_file} \\
        --out standardize.parquet \\
        --overwrite-inchikey \\
        --dedup \\
        --dedup-thresh 0.7 \\
        --export-csv \\
        --engine auto
    """
}