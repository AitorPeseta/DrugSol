#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Final Report
========================================================================================
    Description:
    Generates comprehensive visualization plots and metrics summary for model
    performance evaluation. Supports both standard evaluation and cross-validation
    consensus modes.
    
    Plot Categories:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Category           │ Plots Generated                                          │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  plots_global/      │ Scatter plots, residuals, bias, error histograms         │
    │                     │ (all data points)                                         │
    │                                                                                 │
    │  plots_physio/      │ Same plots filtered for physiological temperature        │
    │                     │ range (35-38°C)                                           │
    │                                                                                 │
    │  plots_classification/ │ ROC curves, PR curves, confusion matrices             │
    │                     │ (binary classification at threshold = -4.0 logS)         │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Classification Threshold:
    - logS > -4.0: Soluble (positive class)
    - logS ≤ -4.0: Insoluble (negative class)
    
    Modes:
    - standard: Shows train (OOF) vs test comparison
    - cv: Shows cross-validation consensus without train/test split
    
    Input:
    - Level-0 predictions
    - Blend/Stack predictions
    - Original test metadata
    - OOF predictions (for train comparison)
    
    Output:
    - Visualization plots
    - Metrics summary CSV and JSON
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: FINAL_REPORT
========================================================================================
*/

process final_report {
    tag "Generate Final Report"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/final_report/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id),
              path(level0_parquet),
              path(blend_parquet),
              path(stack_parquet_opt),
              path(test_data_original, stageAs: "input_metadata.parquet"),
              path(oof_file, stageAs: "input_oof.parquet")
        val  outdir_val
        path script_py
        val  is_cv_mode

    output:
        tuple val(meta_id), path("plots_global/*.png"),         emit: PLOTS_GLOBAL
        tuple val(meta_id), path("plots_physio/*.png"),         emit: PLOTS_PHYSIO, optional: true
        tuple val(meta_id), path("plots_classification/*.png"), emit: PLOTS_CLASS, optional: true
        tuple val(meta_id), path("metrics_summary.csv"),        emit: SUMMARY_CSV
        tuple val(meta_id), path("metrics_global.json"),        emit: METRICS_JSON

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        def target_col = params.report_target_col ?: 'logS'
        def id_col = params.report_id_col ?: 'row_uid'
        def temp_col = params.report_temp_col ?: 'temp_C'
        
    """
    mkdir -p plots_global
    mkdir -p plots_physio
    mkdir -p plots_classification

    # Stack argument (optional)
    STACK_ARG=""
    if [ -n "${stack_parquet_opt}" ] && [ -f "${stack_parquet_opt}" ]; then
        STACK_ARG="--stack ${stack_parquet_opt}"
    fi

    # Mode argument (cv or standard)
    MODE_ARG=""
    if [ "${is_cv_mode}" == "true" ]; then
        MODE_ARG="--mode cv"
    fi

    python ${script_py} \\
        --level0 "${level0_parquet}" \\
        --blend "${blend_parquet}" \\
        \$STACK_ARG \\
        \$MODE_ARG \\
        --metadata "${test_data_original}" \\
        --oof "${oof_file}" \\
        --target "${target_col}" \\
        --id-col "${id_col}" \\
        --temp-col "${temp_col}" \\
        --outdir .
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
