nextflow.enable.dsl = 2

process measurement_counts_histogram {
    tag "EDA: Measurements per Molecule"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/analysis", mode: 'copy', overwrite: true

    input:
        path file      // Parquet file
        val  outdir
        path script_py      // Python script

    output:
        path "meas_dual_combined", emit: COUNTS_DIR
    
    script:
    """
    python ${script_py} \\
        --input "${file}" \\
        --id_col smiles_neutral \\
        --outdir meas_dual_combined
    """
}