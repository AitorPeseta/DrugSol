#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Train OOF Physics
========================================================================================
    Description:
    Trains a physics-informed baseline model using interpretable molecular features
    and thermodynamic principles. This model serves as a baseline and provides
    complementary predictions for ensemble stacking.
    
    Physics-Based Features:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Feature     │ Physical Meaning                                                │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  LogP        │ Octanol-water partition coefficient (hydrophobicity)            │
    │  TPSA        │ Topological polar surface area (polarity)                       │
    │  MW          │ Molecular weight (size)                                          │
    │  1/T         │ Inverse temperature (Van't Hoff thermodynamic dependency)       │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Van't Hoff Equation Background:
    The temperature dependency of solubility follows the Van't Hoff equation:
    
        ln(S) = -ΔH°/R × (1/T) + ΔS°/R
    
    By including 1/T as a feature, the model can learn the enthalpy-driven
    component of solubility temperature dependency.

    Final Equation:
    logS = β₀ + β₁·(1/T) + β₂·LogP + β₃·TPSA + β₄·MW
    
    Model: Ridge Regression with automatic alpha selection (RidgeCV)
    
    Input:
    - Training data with RDKit features (LogP, TPSA, MW) and temperature
    - Pre-computed fold assignments
    
    Output:
    - OOF predictions for ensemble stacking
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: TRAIN_OOF_PHYSICS
========================================================================================
*/

process train_oof_physics {
    tag "Train Physics Baseline"
    label 'cpu_small'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/training/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(folds_file)  // Training data and folds
        val  outdir_val                                          // Output directory
        path script_py                                           // Python script

    output:
        tuple val(meta_id), path("oof_physics/oof_physics.parquet"),        emit: OOF_PHYSICS
        tuple val(meta_id), path("oof_physics/metrics_oof_physics.json"),   emit: METRICS

    script:
    """
    python ${script_py} \\
        --train "${train_file}" \\
        --folds-file "${folds_file}" \\
        --id-col "row_uid" \\
        --target "logS" \\
        --save-dir oof_physics
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
