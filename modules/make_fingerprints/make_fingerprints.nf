nextflow.enable.dsl = 2

process make_fingerprints {
    tag "ECFP4 & Butina Clustering #${iter_id}"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:  
        tuple val(iter_id), path(input_file)     // Input parquet file
        val  outdir_val
        path script_py      // Python script
        val  name_prefix    // e.g. "cluster_ecfp4_0p7"

    output:
        tuple val(iter_id), path("${name_prefix}_fingerprint.parquet"), emit: out

    script:
    """
    # Generate ECFP4 fingerprints (bits) and perform Butina clustering
    # Used for stratified splitting to prevent data leakage
    
    python ${script_py} \\
        --input ${input_file} \\
        --out-parquet "${name_prefix}_fingerprint.parquet" \\
        --smiles-col smiles_neutral \\
        --n-bits 2048 \\
        --radius 2 \\
        --cluster-cutoff 0.7 \\
        --save-csv
    """
}