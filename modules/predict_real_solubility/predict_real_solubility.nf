nextflow.enable.dsl = 2

process predict_real_solubility {
    tag "Physiological Solubility pH ${ph_val}"
    label 'cpu_small'
    
    conda "${params.conda_env_train}"
    
    publishDir "${params.outdir}/predictions", mode: 'copy', overwrite: true

    input:
        path predictions_csv
        val outdir
        path script_py
        val ph_val

    output:
        path "predictions_physio_pH${ph_val}.csv", emit: csv

    script:
    """
    python ${script_py} \\
        --input ${predictions_csv} \\
        --output predictions_physio_pH${ph_val}.csv \\
        --ph ${ph_val}
    """
}