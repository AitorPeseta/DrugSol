nextflow.enable.dsl = 2

process align_feature_columns {
    tag "Align Test to Train"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/aligned", mode: 'copy', overwrite: true

    input:
        tuple path(train_file), path(test_file) // Tuple ensures we have the pair
        val  outdir_val
        path script_py    // Python script

    output:
        path "features_test_aligned.parquet", emit: out

    script:
    """
    # Force Test dataset to have exactly the same columns (and order) as Train
    # Missing columns in Test are filled with 0 (for one-hot) or NaN (for numerical)
    
    python ${script_py} \\
        --train ${train_file} \\
        --test ${test_file} \\
        --out "features_test_aligned.parquet"
    """
}