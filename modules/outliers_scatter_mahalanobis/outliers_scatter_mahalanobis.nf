nextflow.enable.dsl = 2

process outliers_scatter_mahalanobis {
    tag "EDA: Scatter & Outliers"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/analysis/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train), path(test)  // Train and Test parquet files
        val  outdir
        path script_py             // Python script

    output:
        tuple val(meta_id), path("out_viz"), emit: OUTLIER_DIR
    
    script:
    """
    # Visualize Data Space using PCA and Mahalanobis distance
    # Useful to check if Test set is within the domain of applicability of Train
    
    python ${script_py} \\
        --train "${train}" \\
        --test "${test}" \\
        --only_cols logS temp_C qed weight \\
        --outlier_col is_outlier \\
        --basis combined \\
        --outdir out_viz
    """
}