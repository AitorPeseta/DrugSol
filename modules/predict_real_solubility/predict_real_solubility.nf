#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Predict Real Solubility
========================================================================================
    Description:
    Calculates effective solubility at physiological pH from model predictions
    using thermodynamic corrections based on ionization state.
    
    Scientific Background:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  The model predicts Sw (experimental solubility at saturation pH).             │
    │  To get solubility at a different pH, we account for ionization:               │
    │                                                                                 │
    │  1. Calculate pH_sat (pH at saturation, where Sw was measured)                 │
    │  2. Calculate S0 (intrinsic solubility of neutral form)                        │
    │  3. Calculate fraction of neutral species at target pH                         │
    │  4. Seff = S0 / f_neutral                                                      │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Henderson-Hasselbalch Equations:
    - Acid:       f_neutral = 1 / (1 + 10^(pH - pKa))
    - Base:       f_neutral = 1 / (1 + 10^(pKa - pH))
    - Zwitterion: f_neutral = 1 / (1 + 10^(pKa_acid - pH) + 10^(pH - pKa_base))
    
    pKa Prediction:
    Uses external API (MolGpKa at xundrug.cn) for pKa estimation.
    API failures return original prediction unchanged.
    
    Input:
    - CSV with 'smiles' and 'predicted_logS' columns
    - Target pH value (default: 7.4 for physiological conditions)
    
    Output:
    - predictions_physio_pH{X}.csv: Enhanced predictions with:
        - pka_acids: List of acidic pKa values
        - pka_bases: List of basic pKa values  
        - pH_sat_calculated: Estimated saturation pH
        - logSeff_pH{X}: Effective solubility at target pH
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: PREDICT_REAL_SOLUBILITY
========================================================================================
*/

process predict_real_solubility {
    tag "Physiological Solubility pH ${ph_val}"
    label 'cpu_small'
    
    conda "${params.conda_env_train}"
    
    publishDir "${params.outdir}/predictions", mode: 'copy', overwrite: true

    input:
        path predictions_csv
        val  outdir
        path script_py
        val  ph_val

    output:
        path "predictions_physio_pH${ph_val}.csv", emit: csv

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail
    
    python ${script_py} \\
        --input ${predictions_csv} \\
        --output predictions_physio_pH${ph_val}.csv \\
        --ph ${ph_val}
    """
}

/*
========================================================================================
    THE END
========================================================================================
*/
