nextflow.enable.dsl = 2

process make_fingerprints {
    tag "ECFP4 & Butina Clustering"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data", mode: 'copy', overwrite: true

    input:  
        path input_file     // Input parquet file
        val  outdir_val
        path script_py      // Python script
        val  name_prefix    // e.g. "cluster_ecfp4_0p7"

    output:
        path "${name_prefix}_fingerprint.parquet", emit: out

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