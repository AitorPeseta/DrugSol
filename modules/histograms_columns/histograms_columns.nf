nextflow.enable.dsl = 2

process histograms_columns {
    tag "EDA: Histograms"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/analysis", mode: 'copy', overwrite: true

    input:
        tuple path(train), path(test)  // Train and test parquet files
        val  outdir
        path  script_py               // Python script

    output:
        path "hist_out", emit: HIST_DIR
    
    script:
    """
    # Generate comparative histograms for key features
    # We check logS (Target), temp_C (Control), and qed/weight (New features)
    
    python ${script_py} \\
        --train "${train}" \\
        --test "${test}" \\
        --cols logS temp_C qed weight \\
        --round-step 0.5 \\
        --overlay \\
        --outdir hist_out
    """
}