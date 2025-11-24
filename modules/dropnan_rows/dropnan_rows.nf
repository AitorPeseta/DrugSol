nextflow.enable.dsl = 2

process dropnan_rows {
    tag "Clean Rows (${mode})"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/final", mode: 'copy', overwrite: true

    input:
        path source_file // Parquet file to clean
        val  outdir_val
        path script_py   // Python script
        val  name_out    // e.g. "final_train_gbm"
        val  mode        // "train" or "test"

    output:
        path "${name_out}.parquet", emit: out

    script:
    """
    # Final Data Cleaning: Drop rows with NaNs in critical columns
    
    python ${script_py} \\
        --input "${source_file}" \\
        --output "${name_out}.parquet" \\
        --mode "${mode}" \\
        --save_csv
    """
}