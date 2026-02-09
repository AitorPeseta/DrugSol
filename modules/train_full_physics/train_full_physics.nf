#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Train Full Physics
========================================================================================
    Description:
    Trains the final physics-informed Ridge Regression model on the complete dataset.
    This model uses interpretable physicochemical features based on thermodynamic
    principles.
    
    Physics-Based Features:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Feature     │ Physical Meaning                                                │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  LogP        │ Octanol-water partition coefficient (hydrophobicity)            │
    │  TPSA        │ Topological polar surface area (polarity)                       │
    │  MW          │ Molecular weight (size)                                          │
    │  1/T         │ Inverse temperature (thermodynamic term)                        │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Output Formats:
    - PKL: Scikit-learn Pipeline (StandardScaler + Ridge) for Python inference
    - JSON: De-standardized raw weights for portable inference without sklearn
    
    The JSON format allows inference in any language using:
        logS = intercept + coef_logP*logP + coef_TPSA*TPSA + coef_MW*MW + coef_invT*(1000/T_kelvin)
    
    Input:
    - Full training data with RDKit features
    
    Output:
    - Sklearn pipeline pickle
    - JSON with raw coefficients
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: TRAIN_FULL_PHYSICS
========================================================================================
*/

process train_full_physics {
    tag "Train Full Physics Baseline"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/training/${meta_id}/models_physics", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file)  // Training data with RDKit features
        val  outdir_val                        // Output directory
        path script_py                         // Python script

    output:
        tuple val(meta_id), path("models_physics/physics_model.json"), emit: MODEL_JSON
        tuple val(meta_id), path("models_physics/physics_ridge.pkl"),  emit: MODEL_PKL
        tuple val(meta_id), path("models_physics"),                    emit: PHYSICS_DIR

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        def target_col = params.full_physics_target_col ?: 'logS'
        
    """
    python ${script_py} \\
        --train "${train_file}" \\
        --target ${target_col} \\
        --save-dir models_physics
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
