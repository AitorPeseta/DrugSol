process balance_dataset {
    tag "Balance Iter #${iter_id}"
    label 'cpu_small'
    conda "${baseDir}/envs/drugsol-data.yml"

    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:
        tuple val(iter_id), path(input_parquet) 
        path script_py
        val seed_base

    output:
        tuple val(iter_id), path("balanced_${iter_id}.parquet"), emit: balanced_data

    script:
    """
    # Calculamos semilla dinámica: Base + ID Iteración
    # Iter 1 -> Seed 43, Iter 2 -> Seed 44...
    CURRENT_SEED=\$(( ${seed_base} + ${iter_id} ))

    echo "Running Balance Iteration ${iter_id} with Seed \$CURRENT_SEED"

    python ${script_py} \\
        --input "${input_parquet}" \\
        --output "balanced_${iter_id}.parquet" \\
        --limit 100 \\
        --bin-size 0.2 \\
        --seed \$CURRENT_SEED
    """
}