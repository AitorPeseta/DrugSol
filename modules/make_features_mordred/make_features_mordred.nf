#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Make Features Mordred
========================================================================================
    Description:
    Calculates comprehensive molecular descriptors using the Mordred library.
    Mordred computes ~1800 2D descriptors and optionally ~200 3D descriptors,
    providing a rich feature set for gradient boosting models.
    
    Features Computed:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Category              │ Count  │ Description                                  │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Mordred 2D            │ ~1600  │ Constitutional, topological, connectivity    │
    │  Mordred 3D (optional) │ ~200   │ Geometry, surface area, shape (slow)         │
    │  RDKit LogP            │ 1      │ Crippen partition coefficient                │
    │  Solvent One-Hot       │ ~5     │ One-hot encoding of solvent type             │
    │  InChIKey Hash         │ 128    │ Hashed scaffold features for grouping        │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    3D Descriptor Calculation:
    - Generates 3D conformer using ETKDG algorithm
    - Optimizes geometry with UFF or MMFF force field
    - Skips molecules exceeding max_atoms_3d threshold
    - Significantly slower than 2D-only calculation
    
    Input:
    - Parquet file with SMILES and metadata columns
    
    Output:
    - Parquet file with original columns plus ~1800 descriptor columns
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: MAKE_FEATURES_MORDRED
========================================================================================
*/

process make_features_mordred {
    tag "Mordred Descriptors (${dataset_name})"
    label 'cpu_medium'
    
    conda "${params.conda_env_data}"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/features", mode: 'copy', overwrite: true
    
    input:
        tuple val(meta_id), path(input_file)  // Input parquet with SMILES
        val  outdir_val                        // Output directory (for logging)
        path script_py                         // Python script: make_features_mordred.py
        val  dataset_name                      // "train" or "test" (for output naming)

    output:
        tuple val(meta_id), path("${dataset_name}_mordred_featured.parquet"), emit: out

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
    
        // Number of hash bins for InChIKey features
        def hash_bins = params.mordred_hash_bins ?: 128
        
        // 3D descriptor settings
        def include_3d = params.mordred_include_3d ? '--include_3d' : ''
        def force_field = params.mordred_force_field ?: 'uff'
        def max_atoms_3d = params.mordred_max_atoms_3d ?: 200
      
        
    """
    python ${script_py} \\
        --input "${input_file}" \\
        --out_parquet "${dataset_name}_mordred_featured.parquet" \\
        --smiles_source "neutral" \\
        --keep-smiles \\
        --inchikey_col "cluster_ecfp4_0p7" \\
        --ik14_hash_bins ${hash_bins} \\
        --keep_inchikey_as_group \\
        --nproc ${task.cpus} \\
        ${include_3d} \\
        --ff ${force_field} \\
        --max_atoms_3d ${max_atoms_3d}
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
