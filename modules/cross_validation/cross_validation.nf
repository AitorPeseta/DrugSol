#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Cross Validation
========================================================================================
    Description:
    Aggregates results from multiple cross-validation splits to compute consensus
    metrics and predictions. Produces combined outputs for final reporting.
    
    Aggregation Strategy:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Consensus Calculation                                                          │
    │                                                                                 │
    │  For each molecule that appears in multiple CV splits:                         │
    │  1. Collect all predictions from different folds                               │
    │  2. Average predictions to get consensus estimate                              │
    │  3. Calculate metrics on consensus predictions                                  │
    │                                                                                 │
    │  This provides a robust estimate that averages over fold variability.          │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Output Files:
    - test_level0.parquet: Consensus Level-0 predictions
    - test_blend.parquet: Consensus blended predictions
    - test_stack.parquet: Consensus stacked predictions (optional)
    - metrics_cv_consensus.json: Metrics on consensus predictions
    - best_strategy.txt: Winner (blend or stack)
    
    Input:
    - Level-0 predictions from all CV splits
    - Blended predictions from all CV splits
    - Stacked predictions from all CV splits (optional)
    - Metrics JSON files from all splits
    
    Output:
    - Aggregated consensus predictions and metrics
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: CROSS_VALIDATION
========================================================================================
*/

process cross_validation {
    tag "CV Aggregation"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/cross_validation", mode: 'copy', overwrite: true

    input:
        path level0_files, stageAs: "level0_*.parquet"  // Level-0 predictions
        path blend_files,  stageAs: "blend_*.parquet"   // Blend predictions
        path stack_files,  stageAs: "stack_*.parquet"   // Stack predictions (optional)
        path metrics,      stageAs: "metrics_*.json"    // Metrics from each split
        path script_py                                   // Python script

    output:
        tuple val("cross_validation"), path("cv_out/test_level0.parquet"), emit: LEVEL0
        tuple val("cross_validation"), path("cv_out/test_blend.parquet"),  emit: BLEND
        tuple val("cross_validation"), path("cv_out/test_stack.parquet"),  emit: STACK, optional: true
        path "best_strategy.txt", emit: BEST_STRATEGY
        path "cv_out/metrics_cv_consensus.json", emit: METRICS

    script:
    """
    mkdir -p cv_out
    
    # Handle optional stack files
    STACK_ARGS=""
    if ls stack_*.parquet >/dev/null 2>&1; then
        STACK_ARGS="--stack-files stack_*.parquet"
    fi

    python ${script_py} \\
        --inputs metrics_*.json \\
        --level0-files level0_*.parquet \\
        --blend-files blend_*.parquet \\
        \$STACK_ARGS \\
        --outdir cv_out
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
