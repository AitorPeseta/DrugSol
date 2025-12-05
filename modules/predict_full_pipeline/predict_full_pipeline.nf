nextflow.enable.dsl = 2

process predict_full_pipeline {
    tag "Inference"
    label 'cpu_medium'
    
    conda "${baseDir}/envs/drugsol-train.yml"

    publishDir "${params.outdir}/predictions", mode: 'copy'

    input:
    path test_gbm     // Features Mordred (parquet)
    path test_smiles  // Features RDKit + SMILES (parquet)
    path final_prod   // Carpeta FINAL_PRODUCT
    path script_py    // predict_full_pipeline.py

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