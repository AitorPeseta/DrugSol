nextflow.enable.dsl = 2

process final_report {
    tag "Generate Final Report"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/final_report/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), 
              path(level0_parquet), 
              path(blend_parquet), 
              path(stack_parquet_opt), 
              path(test_data_original, stageAs: "input_metadata.parquet"), 
              path(oof_file,           stageAs: "input_oof.parquet")
              
        val  outdir_val
        path script_py
        val  is_cv_mode

    output:
        tuple val(meta_id), path("plots_global/*.png"),      emit: PLOTS_GLOBAL
        tuple val(meta_id), path("plots_physio/*.png"),      emit: PLOTS_PHYSIO
        tuple val(meta_id), path("plots_classification/*.png"),  emit: PLOTS_CLASS
        tuple val(meta_id), path("metrics_summary.csv"),     emit: SUMMARY_CSV
        tuple val(meta_id), path("metrics_global.json"),     emit: METRICS_JSON

    script:
    """
    mkdir -p plots_global
    mkdir -p plots_physio
    mkdir -p plots_classification

    # 1. Lógica para Stack (existente)
    STACK_ARG=""
    if [ -n "${stack_parquet_opt}" ] && [ -f "${stack_parquet_opt}" ]; then
        STACK_ARG="--stack ${stack_parquet_opt}"
    fi

    # 2. Lógica para Modo CV (NUEVA)
    # Nextflow convierte el val booleano true/false a cadena "true"/"false" en bash
    MODE_ARG=""
    if [ "${is_cv_mode}" == "true" ]; then
        MODE_ARG="--mode cv"
    fi

    # 3. Ejecución
    python ${script_py} \\
        --level0 "${level0_parquet}" \\
        --blend "${blend_parquet}" \\
        \$STACK_ARG \\
        \$MODE_ARG \\
        --metadata "${test_data_original}" \\
        --oof "${oof_file}" \\
        --target "logS" \\
        --id-col "row_uid" \\
        --temp-col "temp_C" \\
        --outdir .
    """
}