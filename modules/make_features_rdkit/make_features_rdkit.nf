#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Make Features RDKit
========================================================================================
    Description:
    Calculates basic physicochemical descriptors using RDKit. These features are
    used by the Physics baseline model and Graph Neural Networks (Chemprop).
    
    Features Computed:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Feature        │ Description                                                  │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  rdkit__TPSA    │ Topological Polar Surface Area (Å²)                          │
    │  rdkit__logP    │ Crippen LogP (octanol-water partition coefficient)           │
    │  rdkit__MW      │ Molecular Weight (Da)                                        │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    These descriptors are fundamental physicochemical properties that correlate
    with aqueous solubility:
    - TPSA: Higher values indicate more polar molecules (generally more soluble)
    - LogP: Higher values indicate more lipophilic molecules (generally less soluble)
    - MW: Larger molecules tend to have lower solubility
    
    The Physics baseline model uses these features combined with Van't Hoff
    temperature dependency (1/T) for interpretable solubility prediction.
    
    Input:
    - Parquet file with SMILES column
    
    Output:
    - Parquet file with original columns plus 3 RDKit descriptor columns
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: MAKE_FEATURES_RDKIT
========================================================================================
*/

process make_features_rdkit {
    tag "RDKit Descriptors (${dataset_name})"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/prepare_data/${meta_id}/features", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(input_file)  // Input parquet with SMILES
        val  outdir_val                        // Output directory (for logging)
        path script_py                         // Python script: make_features_rdkit.py
        val  dataset_name                      // "train" or "test" (for output naming)

    output:
        tuple val(meta_id), path("${dataset_name}_rdkit_featured.parquet"), emit: out

    script:  
    """
    python ${script_py} \\
        --input "${input_file}" \\
        --out "${dataset_name}_rdkit_featured.parquet" \\
        --smiles-col "smiles_neutral"
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
