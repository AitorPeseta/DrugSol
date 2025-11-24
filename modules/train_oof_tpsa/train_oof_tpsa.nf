nextflow.enable.dsl = 2

process train_oof_tpsa {
    tag "Train Baseline (Ridge)"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/training", mode: 'copy', overwrite: true

    input:
        path train_file    // Training data file
        val  outdir_val
        path script_py     // Python script
        path folds_file    // Cross-validation folds file

    output:
        path "oof_tpsa/oof_tpsa.parquet", emit: OFF_TPSA
        path "oof_tpsa/metrics_oof_tpsa.json", emit: META_TPSA

    script:
    """
    # Train a Ridge Regression Baseline
    # Features: logP, TPSA, MW, 1/Temperature
    
    python ${script_py} \\
        --train "${train_file}" \\
        --folds-file "${folds_file}" \\
        --id-col row_uid \\
        --target logS \\
        --save-dir oof_tpsa
    """
}