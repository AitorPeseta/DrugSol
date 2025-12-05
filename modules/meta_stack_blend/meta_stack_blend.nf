nextflow.enable.dsl = 2

process meta_stack_blend {
    tag "Ensemble Stacking & Blending"
    label 'cpu_small'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training/${meta_id}/ensemble", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(oof_lgbm), path(oof_xgb), path(oof_gnn), path(oof_tpsa)  // OOF prediction files from base models 
        val  outdir_val
        path script_py   // Python script

    output:
        tuple val(meta_id), path("meta_results/blend/weights.json"),      emit: BLEND_WEIGHTS
        tuple val(meta_id), path("meta_results/stack/meta_ridge.pkl"),    emit: STACK_MODEL
        tuple val(meta_id), path("meta_results/oof_predictions.parquet"), emit: OOF_COMBINED
        tuple val(meta_id), path("meta_results/metrics_oof.json"),        emit: METRICS_OOF 

    script:
    """
    # Combines OOF predictions from all base models using Stacking (Ridge) and Blending
    # Automatically selects the best strategy based on RMSE
    
    python ${script_py} \\
        --oof-common "${oof_lgbm}" "${oof_xgb}" "${oof_gnn}" "${oof_tpsa}" \\
        --labels lgbm xgb gnn tpsa \\
        --metric rmse \\
        --save-dir meta_results
    """
}