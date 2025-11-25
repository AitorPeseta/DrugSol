nextflow.enable.dsl = 2

process train_full_chemprop {
    tag "Train Full Chemprop"
    label 'process_gpu' // Request GPU
    accelerator 1, type: 'nvidia'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training", mode: 'copy', overwrite: true

    input:
        path train_file
        val  outdir_val
        path script_py
        path best_params_json

    output:
        path "models_GNN", emit: CHEMPROP_DIR
        path "models_GNN/chemprop_manifest.json", emit: CHEMPROP_MANIFEST

    script:
    """
    export LD_PRELOAD=\$(find "\$PREFIX/lib" -name "libmkl_core.so*" | head -n 1)
    if [ -z "\$LD_PRELOAD" ]; then
        export LD_PRELOAD=\$(find "\$PREFIX/lib" -name "libmkl_intel_lp64.so*" | head -n 1)
    fi
    echo "LD_PRELOAD set to: \$LD_PRELOAD"

    python ${script_py} \\
        --train "${train_file}" \\
        --smiles-col smiles_neutral \\
        --target logS \\
        --best-params "${best_params_json}" \\
        --epochs 40 \\
        --gpu \\
        --save-dir models_GNN
    """
}