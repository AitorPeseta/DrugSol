nextflow.enable.dsl = 2

process make_folds {
    tag "Generate CV Folds"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/training/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file)
        val  outdir_val
        path script_py    // Python script

    output:
        tuple val(meta_id), path("folds.parquet"), emit: out

    script:
    """
    # Generate Cross-Validation folds
    # Strategy: Grouped (by scaffold) & Stratified (by Target + Temp)
    
    python ${script_py} \\
        --input "${train_file}" \\
        --out "folds.parquet" \\
        --id-col row_uid \\
        --group-col cluster_ecfp4_0p7 \\
        --target logS \\
        --strat-mode both \\
        --temp-col temp_C \\
        --temp-step 5 \\
        --bins 5 \\
        --n-splits 5 \\
        --seed 42
    """
}