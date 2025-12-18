nextflow.enable.dsl = 2

process dropnan_rows {
    tag "Clean Rows (${mode})"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/final", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(source_file)
        val  outdir_val
        path script_py   // Python script
        val  name_out    // e.g. "final_train_gbm"
        val  mode        // "train" or "test"
        val  subset

    output:
        tuple val(meta_id), path ("${name_out}.parquet"), emit: out

    script:
    """
    # Final Data Cleaning: Drop rows with NaNs in critical columns
    
    python ${script_py} \\
        --input "${source_file}" \\
        --output "${name_out}.parquet" \\
        --mode "${mode}" \\
        --subset features_only \\
        --save_csv
    """
}