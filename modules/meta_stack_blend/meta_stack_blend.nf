nextflow.enable.dsl = 2

process meta_stack_blend {
    tag "Ensemble Stacking & Blending"
    label 'cpu_small'
    
    // Uses the standard training environment (scikit-learn)
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training/ensemble", mode: 'copy', overwrite: true

    input:
        tuple path(oof_lgbm), path(oof_xgb), path(oof_gnn), path(oof_tpsa)  // OOF prediction files from base models 
        val  outdir_val
        path script_py   // Python script
        val  suffix      // e.g. "_v1"

    output:
        path "meta_results/blend/weights${suffix}.json",      emit: BLEND_WEIGHTS
        path "meta_results/stack/meta_ridge${suffix}.pkl",    emit: STACK_MODEL
        path "meta_results/oof_predictions${suffix}.parquet", emit: OOF_COMBINED
        path "meta_results/metrics_oof${suffix}.json",        emit: METRICS_OOF 

    script:
    """
    # Combines OOF predictions from all base models using Stacking (Ridge) and Blending
    # Automatically selects the best strategy based on RMSE
    
    python ${script_py} \\
        --oof-common "${oof_lgbm}" "${oof_xgb}" "${oof_gnn}" "${oof_tpsa}" \\
        --labels lgbm xgb gnn tpsa \\
        --metric rmse \\
        --suffix "${suffix}" \\
        --save-dir meta_results
    """
}