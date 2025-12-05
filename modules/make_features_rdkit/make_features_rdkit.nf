nextflow.enable.dsl = 2

process make_features_rdkit {
    tag "RDKit Descriptors (${dataset_name})"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    // Publish results. If it's the training set, we might want to keep it as a resource resource
    publishDir "${params.outdir}/prepare_data/${meta_id}/features", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(input_file)
        val  outdir_val
        path script_py         // Python script
        val  dataset_name      // "train" or "test"

    output:
        tuple val(meta_id), path("${dataset_name}_rdkit_featured.parquet"), emit: out

    script:
    """
    # Calculate RDKit basic features
    # No need for 'cp' here, publishDir handles it automatically.
    
    python ${script_py} \\
        --input "${input_file}" \\
        --out "${dataset_name}_rdkit_featured.parquet" \\
        --smiles-col smiles_neutral
    """
}