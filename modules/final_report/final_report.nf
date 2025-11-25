nextflow.enable.dsl = 2

process final_report {
    tag "Generate Analysis Report"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/final_report", mode: 'copy', overwrite: true

    input:
        path level0_parquet             // Prediction
        path blend_parquet
        path stack_parquet_opt
        path test_data_original         // For to take the temperature
        val  outdir_val
        path script_py                  // Script

    output:
        path "plots_global/*.png",      emit: PLOTS_GLOBAL
        path "plots_physio/*.png",      emit: PLOTS_PHYSIO
        path "metrics_global.json",     emit: METRICS_GLOBAL
        path "metrics_physio.json",     emit: METRICS_PHYSIO
        path "metrics_summary.csv",     emit: SUMMARY_CSV

    script:
    """
    # Preparamos carpetas
    mkdir -p plots_global
    mkdir -p plots_physio

    # Construimos argumentos
    FILES_ARGS="--level0 ${level0_parquet} --blend ${blend_parquet}"
    
    if [ -f "${stack_parquet_opt}" ]; then
        FILES_ARGS="\$FILES_ARGS --stack ${stack_parquet_opt}"
    fi
    
    python ${script_py} \\
        \$FILES_ARGS \\
        --metadata "${test_data_original}" \\
        --target "logS" \\
        --id-col "row_uid" \\
        --temp-col "temp_C" \\
        --bin-step 0.2 \\
        --outdir .
    """
}