#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Publish Resources
========================================================================================
    Description:
    Publishes reference datasets to the resources directory for use in
    execution (inference) mode. These files serve as templates for
    feature alignment when processing new molecules.
    
    Purpose:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  During RESEARCH mode:                                                          │
    │  - Features are computed and filtered based on training data                    │
    │  - The final feature sets define the model's input schema                       │
    │                                                                                 │
    │  During EXECUTION mode:                                                         │
    │  - New molecules must have the SAME features as training data                   │
    │  - These reference files ensure column alignment                                │
    │  - Missing columns are filled with zeros/defaults                               │
    │  - Extra columns are dropped                                                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Published Files:
    - train_features_mordred_filtered.parquet:
        Mordred descriptors after correlation filtering and importance selection.
        Used by GBM models (XGBoost, LightGBM, CatBoost).
        
    - train_rdkit_featured.parquet:
        RDKit descriptors for Chemprop and physics baseline.
        Simpler feature set for graph neural network inputs.
    
    Location:
    Files are copied to ${baseDir}/resources/ for persistent storage
    across pipeline runs.
    
    Input:
    - Full Mordred feature parquet from training
    - Full RDKit feature parquet from training
    
    Output:
    - Reference parquet files in resources directory
    
    Note:
    This process runs at the end of RESEARCH mode to prepare for future
    EXECUTION runs without needing to retrain.
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: PUBLISH_RESOURCES
========================================================================================
*/

process publish_resources {
    tag "Publishing Execution Resources"
    label 'cpu_small'
    
    // Publish to resources directory (persistent across runs)
    publishDir "${baseDir}/resources", mode: 'copy', overwrite: true

    input:
        path full_mordred, stageAs: 'mordred_source.parquet'
        path full_rdkit,   stageAs: 'rdkit_source.parquet'

    output:
        path "train_features_mordred_filtered.parquet", emit: mordred_ref
        path "train_rdkit_featured.parquet",            emit: rdkit_ref

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail
    
    # Copy and rename to standard reference names
    cp ${full_mordred} train_features_mordred_filtered.parquet
    cp ${full_rdkit}   train_rdkit_featured.parquet
    
    echo "[Publish] Reference files created for execution mode"
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
