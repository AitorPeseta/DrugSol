#!/usr/bin/env nextflow
/*
========================================================================================
    Module: Make Embeddings ChemBERTa
========================================================================================
    Description:
    Generates molecular embeddings using ChemBERTa, a transformer model pre-trained
    on 77 million molecules from PubChem. These embeddings capture learned chemical
    knowledge that complements traditional molecular descriptors.

    Instead of fine-tuning ChemBERTa for solubility prediction directly,          
    we use it as a "chemical knowledge extractor":                                
                                                                                    
    1. Pass SMILES through pre-trained ChemBERTa                                   
    2. Extract the [CLS] token embedding (768 dimensions)                          
    3. Add these 768 features to the GBM input (alongside Mordred)                 
    4. Let gradient boosting find patterns in the embeddings                       
                                                                                    
    Benefits:                                                                      
    - Injects "chemical intuition" from 77M molecules                              
    - No expensive fine-tuning required                                            
    
    Model Options:
    - seyonec/ChemBERTa-zinc-base-v1     : Trained on ZINC (default)
    - seyonec/ChemBERTa-zinc250k-v1     : Trained on ZINC250k subset
    - DeepChem/ChemBERTa-77M-MLM        : Trained on PubChem 77M (largest)
    - DeepChem/ChemBERTa-77M-MTR        : Multi-task regression variant
    
    Output Features:
    - bert_0, bert_1, ..., bert_767: 768-dimensional embedding vector
    
    Input:
    - Parquet file with SMILES column
    
    Output:
    - Parquet file with original columns + 768 embedding columns
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: MAKE_EMBEDDINGS_CHEMBERTA
========================================================================================
*/

process make_embeddings_chemberta {
    tag "ChemBERTa Embeddings #${iter_id}"
    label 'process_gpu'
    
    conda "${baseDir}/envs/drugsol-bert.yml"  // Separate env with transformers
    
    publishDir "${params.outdir}/prepare_data/iter_${iter_id}", mode: 'copy', overwrite: true

    input:
        tuple val(iter_id), path(input_file)  // Input parquet with SMILES
        val  outdir_val                        // Output directory (for logging)
        path script_py                         // Python script: make_embeddings_chemberta.py
        val  split_name                        // "train" or "test" (for output naming)

    output:
        tuple val(iter_id), path("${split_name}_chemberta_embeddings.parquet"), emit: out

    script:
        // ---------------------------------------------------------------------------
        // Configurable Parameters
        // ---------------------------------------------------------------------------
        
        // ChemBERTa model to use (HuggingFace model ID)
        // Options:
        //   - "seyonec/ChemBERTa-zinc-base-v1" (default, good balance)
        //   - "DeepChem/ChemBERTa-77M-MLM" (largest, best quality)
        //   - "DeepChem/ChemBERTa-77M-MTR" (multi-task variant)
        def model_name = params.chemberta_model ?: 'seyonec/ChemBERTa-zinc-base-v1'
        
        // Batch size for inference (adjust based on GPU memory)
        // CPU: 16-32, GPU: 64-256
        def batch_size = params.chemberta_batch_size ?: 32
   
        // Pooling strategy: "cls" (default), "mean", or "max"
        def pooling = params.chemberta_pooling ?: 'cls'
        
        // Whether to use GPU (auto-detect if not specified)
        def use_gpu = params.chemberta_use_gpu != null ? params.chemberta_use_gpu : 'auto'
        
    """
    #!/usr/bin/env bash
    set -euo pipefail
    
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

/*
========================================================================================
    THE END
========================================================================================
*/
