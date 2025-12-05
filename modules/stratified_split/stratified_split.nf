nextflow.enable.dsl = 2

process stratified_split {
    tag "Stratified Scaffold Split"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/splits", mode: 'copy', overwrite: true

    input:
        path source_file  // Parquet file to split
        val  outdir_val
        path script_py    // Python script
        val n_iters
        val seed

    output:
        path "split_*" , emit: splits

    script:
    """
    # Performs a Group Stratified Split.
    # Groups = Scaffolds (Butina Clusters) to prevent structure leakage.
    # Strata = Solvent + Temperature bins to ensure balanced conditions.
    
    python ${script_py} \\
        --input ${source_file} \\
        --group-col "cluster_ecfp4_0p7" \\
        --temp-col "temp_C" \\
        --temp-step 5 \\
        --test-size 0.2 \\
        --seed 42 \\
        --min-groups-per-class 2 \\
        --seed ${seed} \\
        --n-splits ${n_iters} \\
        --outdir .
    """
}