#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Build Final Ensemble
========================================================================================
    Description:
    Packages the final production model by combining trained base models and
    training the meta-learner on OOF predictions. Creates a deployable artifact
    containing all components needed for inference.
    
    Final Product Structure:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  drugsol_model/                                                                 │
    │  ├── base_models/                                                               │
    │  │   ├── gbm/          # XGBoost, LightGBM, CatBoost models                    │
    │  │   ├── gnn/          # Chemprop D-MPNN checkpoint                            │
    │  │   └── physics/      # Physics-informed Ridge model                          │
    │  ├── weights.json      # Blending weights (if strategy=blend)                  │
    │  ├── meta_ridge.pkl    # Stacking meta-model (if strategy=stack)               │
    │  └── model_card.json   # Model metadata and version info                       │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Meta-Learning Strategies:
    - Blend: Non-negative least squares (NNLS) weights, normalized to sum to 1
    - Stack: Ridge regression meta-model with cross-validated alpha
    
    Input:
    - Winning strategy (blend or stack)
    - OOF predictions from all base models
    - Trained base models (GBM, GNN, Physics)
    - Training data for meta-learner fitting
    
    Output:
    - Complete deployable model package
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: BUILD_FINAL_ENSEMBLE
========================================================================================
*/

process build_final_ensemble {
    tag "Build Final Product"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-train.yml"

    publishDir "${params.outdir}/final_product", mode: 'copy', overwrite: true

    input:
        val  strategy                                    // Winning strategy (blend/stack)
        path oof_preds_all, stageAs: "oof_part_*.parquet"  // OOF predictions
        path gbm_models, stageAs: "input_gbm"            // GBM models directory
        path gnn_models, stageAs: "input_gnn"            // GNN models directory
        path physics_model, stageAs: "input_physics"     // Physics model (file or dir)
        val  outdir                                      // Output directory
        path script_py                                   // Python script
        path train_file                                  // Training data for meta-learner

    output:
        path "drugsol_model/*", emit: MODEL_DIR

    script:
    """
    echo "Building final ensemble with strategy: ${strategy}"
    
    python ${script_py} \\
        --strategy ${strategy} \\
        --oof-files oof_part_*.parquet \\
        --train-file ${train_file} \\
        --gbm-dir input_gbm \\
        --gnn-dir input_gnn \\
        --physics-model input_physics \\
        --save-dir drugsol_model
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
