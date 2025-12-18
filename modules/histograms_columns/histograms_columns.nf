nextflow.enable.dsl = 2

process histograms_columns {
    tag "EDA: Histograms"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/analysis/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train), path(test)  // Train and test parquet files
        val  outdir
        path  script_py               // Python script

    output:
        tuple val(meta_id), path("hist_out"), emit: HIST_DIR
    
    script:
    """
    # Generate comparative histograms for key features
    # We check logS (Target), temp_C (Control), and qed/weight (New features)
    export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"

    python ${script_py} \\
        --train "${train}" \\
        --test "${test}" \\
        --cols logS temp_C qed MW \\
        --round-step 0.5 \\
        --overlay \\
        --outdir hist_out
    """
}