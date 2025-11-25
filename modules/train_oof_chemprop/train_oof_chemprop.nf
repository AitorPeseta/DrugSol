nextflow.enable.dsl = 2

process train_oof_chemprop {
    tag "Train Chemprop (OOF)"
    label 'process_gpu' 
    accelerator 1, type: 'nvidia'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training", mode: 'copy', overwrite: true

    input:
        path train_file
        val  outdir_val
        path script_py
        path folds_file

    output:
        path "oof_gnn/chemprop.parquet", emit: OFF_GNN
        path "oof_gnn/chemprop_best_params.json", emit: BEST_GNN
        path "oof_gnn/metrics_oof_chemprop.json", emit: META_GNN

    script:
    """
    export LD_PRELOAD=\$(find "\$PREFIX" -name "libmkl_core.so*" | head -n 1)
    if [ -z "\$LD_PRELOAD" ]; then
        export LD_PRELOAD=\$(find "\$PREFIX" -name "libmkl_intel_lp64.so*" | head -n 1)
    fi
    echo "LD_PRELOAD set to: \$LD_PRELOAD"
    
    python ${script_py} \\
        --train "${train_file}" \\
        --folds "${folds_file}" \\
        --smiles-col smiles_neutral \\
        --id-col row_uid \\
        --target logS \\
        --weight-col sw_temp37 \\
        --tune-trials 15 \\
        --epochs 40 \\
        --tune-pruner asha \\
        --asha-rungs 5 10 15 \\
        --save-dir ./oof_gnn
    """
}