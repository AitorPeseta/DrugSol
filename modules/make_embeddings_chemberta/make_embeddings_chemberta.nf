#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Make Embeddings ChemBERTa
========================================================================================
    Generates molecular embeddings using ChemBERTa transformer model.
    Uses pre-created conda environment for reliability.
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

process make_embeddings_chemberta {
    tag "ChemBERTa Embeddings #${iter_id}"
    label 'process_gpu'
    
    // Use pre-created environment path instead of yml file
    conda "${params.conda_env_bert ?: "${baseDir}/envs/conda_cache/drugsol-bert"}"
    
    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:
        tuple val(iter_id), path(input_file)
        val  outdir_val
        path script_py
        val  split_name

    output:
        tuple val(iter_id), path("${split_name}_chemberta_embeddings.parquet"), emit: out

    script:
        def model_name = params.chemberta_model ?: 'seyonec/ChemBERTa-zinc-base-v1'
        def batch_size = params.chemberta_batch_size ?: 32
        def pooling = params.chemberta_pooling ?: 'cls'
        def use_gpu = params.chemberta_use_gpu != null ? params.chemberta_use_gpu : 'auto'
        
    """
    #!/usr/bin/env bash
    set -euo pipefail
    
    # Verify environment
    echo "[ChemBERTa] Verifying environment..."
    python -c "import torch, transformers, pandas; print('Environment OK')"
    
    # Run embedding generation
    python ${script_py} \\
        --input "${input_file}" \\
        --output "${split_name}_chemberta_embeddings.parquet" \\
        --smiles-col "smiles_neutral" \\
        --model-name "${model_name}" \\
        --batch-size ${batch_size} \\
        --embed-prefix "bert_" \\
        --pooling "${pooling}" \\
        --device "${use_gpu}"
    """
}
