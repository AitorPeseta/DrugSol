#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Predict Full Pipeline
========================================================================================
    Description:
    Production inference pipeline that generates predictions for new molecules
    using the packaged final product model. Handles all base model predictions
    and ensemble combination automatically.
    
    Inference Flow:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Input Data                                                                     │
    │  ├── test_gbm.parquet   (Mordred features for GBM models)                      │
    │  └── test_smiles.parquet (SMILES + RDKit features for GNN/Physics)             │
    │                                                                                 │
    │  Base Model Predictions                                                         │
    │  ├── XGBoost    ─┐                                                             │
    │  ├── LightGBM   ─┼─→ Meta-Learner → Final Prediction                          │
    │  ├── CatBoost   ─┤                                                             │
    │  ├── Chemprop   ─┤                                                             │
    │  └── Physics    ─┘                                                             │
    │                                                                                 │
    │  Output: predictions.csv                                                        │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    The module automatically:
    - Loads all base models from the final product package
    - Generates predictions from each model
    - Combines predictions using the trained meta-learner
    - Handles missing models gracefully
    
    Input:
    - Test data with Mordred features (for GBM)
    - Test data with SMILES (for GNN and Physics)
    - Final product directory
    
    Output:
    - CSV file with final predictions
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: PREDICT_FULL_PIPELINE
========================================================================================
*/

process predict_full_pipeline {
    tag "Inference"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-train.yml"

    publishDir "${params.outdir}/predictions", mode: 'copy'

    input:
        path test_gbm      // Mordred features parquet
        path test_smiles   // SMILES + RDKit features parquet
        path final_prod    // Final product directory
        path script_py     // Python script

    output:
        path "predictions.csv", emit: CSV

    script:
    """
    python3 ${script_py} \\
        --data-gbm ${test_gbm} \\
        --data-gnn ${test_smiles} \\
        --final-product ${final_prod} \\
        --output predictions.csv
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
