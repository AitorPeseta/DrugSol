nextflow.enable.dsl = 2

process engineer_features {
    tag "Physics & Chemistry"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data", mode: 'copy', overwrite: true

    input:
        path input_parquet   // Input parquet
        val  outdir_val
        path script_py       // Python script

    output:
        path "engineered_features.parquet", emit: out

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

    rm -r ${baseDir}/bin/__pycache__
    """
}