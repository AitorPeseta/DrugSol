nextflow.enable.dsl = 2

process train_full_tpsa {
    tag "Train Full Baseline (Ridge)"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/training/models_TPSA", mode: 'copy', overwrite: true

    input:
        path train_file   // Training data file
        val  outdir_val
        path script_py    // Python script

    output:
        path "models_TPSA/tpsa_model.json", emit: TPSA_MODEL
        path "models_TPSA/tpsa_phys.pkl",   emit: TPSA_PKL

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