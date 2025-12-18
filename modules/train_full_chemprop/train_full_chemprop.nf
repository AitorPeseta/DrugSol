nextflow.enable.dsl = 2

process train_full_chemprop {
    tag "Train Full Chemprop"
    label 'process_gpu'
    
    conda "${baseDir}/envs/drugsol-train.yml"
    
    publishDir "${params.outdir}/training/${meta_id}", mode: 'copy', overwrite: true

    input:
        tuple val(meta_id), path(train_file), path(best_params_json)
        val  outdir_val
        path script_py

    output:
        tuple val(meta_id), path("models_GNN"), emit: CHEMPROP_DIR
        tuple val(meta_id), path("models_GNN/chemprop_manifest.json"), emit: CHEMPROP_MANIFEST

    script:
    """
    set -euo pipefail

    echo "Buscando librerías MKL en: \$CONDA_PREFIX"
    
    export LD_PRELOAD=\$(find "\$CONDA_PREFIX/lib" -name "libmkl_core.so*" | head -n 1)
    
    if [ -z "\$LD_PRELOAD" ]; then
        # Intento alternativo
        export LD_PRELOAD=\$(find "\$CONDA_PREFIX/lib" -name "libmkl_intel_lp64.so*" | head -n 1)
    fi
    
    echo "LD_PRELOAD set to: \$LD_PRELOAD"
    python ${script_py} \\
        --train "${train_file}" \\
        --smiles-col smiles_neutral \\
        --target logS \\
        --best-params "${best_params_json}" \\
        --epochs 40 \\
        --gpu \\
        --weight-col weight \\
        --save-dir models_GNN
    """
}