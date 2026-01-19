nextflow.enable.dsl = 2

process engineer_features {
    tag "Physics & Chemistry #${iter_id}"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:
        tuple val(iter_id), path(input_parquet)   // Input parquet
        val  outdir_val
        path script_py       // Python script

    output:
        tuple val(iter_id), path("engineered_features.parquet"), emit: out

    script:
    """
    # Calculates:
    # 1. Gaussian Weights (based on Temp vs 37C)
    # 2. QED (Drug-likeness)
    # 3. Ionization (Acid/Base counts)
    # 4. pKa (via external API)
    
    python ${script_py} \\
        --in ${input_parquet} \\
        --out engineered_features.parquet \\
        --smiles-col smiles_neutral \\
        --temp-col temp_C \\
        --smarts "${baseDir}/resources/smarts_pattern_ionized.txt" \\
        --pka-api-url "http://xundrug.cn:5001/modules/upload0/" \\
        --pka-token "O05DriqqQLlry9kmpCwms2IJLC0MuLQ7" \\
        --nproc ${task.cpus}
    """
}