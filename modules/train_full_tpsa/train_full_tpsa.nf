nextflow.enable.dsl = 2

process train_full_tpsa {
    tag "Train Full Baseline (Ridge)"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/training/${meta_id}/models_TPSA", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file)
        val  outdir_val
        path script_py    // Python script

    output:
        tuple val(meta_id), path("models_TPSA/tpsa_model.json"), emit: TPSA_MODEL
        tuple val(meta_id), path("models_TPSA/tpsa_phys.pkl"),   emit: TPSA_PKL

    script:
    """
    # Train the final Ridge Regression model on 100% of the data.
    # Exports both a Python pickle (sklearn) and a JSON with raw weights (portable).
    
    python ${script_py} \\
        --train "${train_file}" \\
        --target logS \\
        --save-dir models_TPSA
    """
}