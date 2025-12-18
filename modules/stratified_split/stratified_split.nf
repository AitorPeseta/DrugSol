process stratified_split {
    tag "Split Iter #${iter_id}"
    label 'cpu_small'
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:
        tuple val(iter_id), path(balanced_file)
        val  outdir_base 
        path script_py
        val seed_fixed 

    output:
        tuple val(iter_id), path("train.parquet"), path("test.parquet"), emit: splits

    script:
    """
    # Ejecutamos el script para hacer UN solo split (80/20)
    # Usamos una semilla fija porque la aleatoriedad ya vino del balanceo previo.
    
    python ${script_py} \\
        --input ${balanced_file} \\
        --group-col "cluster_ecfp4_0p7" \\
        --temp-col "temp_C" \\
        --temp-step 5 \\
        --test-size 0.2 \\
        --seed ${seed_fixed} \\
        --outdir . 
    """
}