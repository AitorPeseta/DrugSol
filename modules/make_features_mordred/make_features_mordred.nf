nextflow.enable.dsl = 2

process make_features_mordred {
    tag "Mordred Descriptors (${dataset_name})"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    // Publish results. If it's the training set, we might want to keep it as a resource resource
    publishDir "${params.outdir}/prepare_data/${meta_id}/features", mode: 'copy', overwrite: true
    
    input:
        tuple val(meta_id), path(input_file)
        val  outdir_val
        path script_py     // Python script
        val  dataset_name  // "train" or "test"

    output:
        tuple val(meta_id), path("${dataset_name}_mordred_featured.parquet"), emit: out

    script:
    """
    # Calculate Mordred Descriptors
    # We dynamically pass the number of CPUs allocated to this task
    
    python ${script_py} \\
        --input ${input_file} \\
        --out_parquet "${dataset_name}_mordred_featured.parquet" \\
        --smiles_source neutral \\
        --keep-smiles \\
        --inchikey_col cluster_ecfp4_0p7 \\
        --ik14_hash_bins 128 \\
        --keep_inchikey_as_group \\
        --nproc ${task.cpus} \\
        --include_3d \\
        --ff uff \\
        --max_atoms_3d 200 \\
        --save_csv
    """
}