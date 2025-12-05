nextflow.enable.dsl = 2

process build_final_ensemble {
    tag "Build Final Product"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-train.yml"

    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    input:
        val strategy
        path oof_preds_all, stageAs: "oof_part_*.parquet"
        path gbm_models, stageAs: "input_gbm"
        path gnn_models, stageAs: "input_gnn"
        path tpsa_model, stageAs: "input_tpsa"
        val outdir
        path script_py
        path train_file

    output:
        path "*"

    script:
    """
    echo "Estrategia ganadora: ${strategy}"
    echo "Construyendo ensemble final..."
    
    # El script Python recibirá los nombres nuevos (input_gbm, input_gnn...)
    # Nextflow se encarga de pasar las rutas correctas.
    
    python ${script_py} \\
        --strategy ${strategy} \\
        --oof-files oof_part_*.parquet \\
        --train-file ${train_file} \\
        --gbm-dir input_gbm \\
        --gnn-dir input_gnn \\
        --tpsa-model input_tpsa \\
        --save-dir final_product
    """
}