nextflow.enable.dsl = 2

process filter_features {
    tag "Filter Features (${dataset_name})"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    // Publish Logic:
    publishDir "${params.outdir}/prepare_data/${meta_id}/filtered", mode: 'copy', overwrite: true
    
    input:
        tuple val(meta_id), path(input_file)
        val  outdir_val
        path script_py     // Python script to run
        val  dataset_name  // "train" or "test"

    output:
        tuple val(meta_id), path("${dataset_name}_features_mordred_filtered.parquet"), emit: out
    
    script:
    """
    # Feature Selection Pipeline:
    # 1. Drop Constant/NZV columns
    # 2. Cluster correlated features
    # 3. Select representative (medoid) by LightGBM gain importance
    
    python ${script_py} \\
        --input "${input_file}" \\
        --output "${dataset_name}_features_mordred_filtered.parquet" \\
        --target logS \\
        --mordred-prefix "mordred__" \\
        --corr-thresh 0.99 \\
        --algo lgbm
    """
}