#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
========================================================================================
    DrugSol - Main Workflow Entrypoint
========================================================================================
    Machine learning pipeline for aqueous solubility prediction.
    
    Modes:
    - research: Full training pipeline (data curation → training → validation)
    - execution: Inference on new molecules using trained model
    
    Author: Aitor Olivares Perucho
    Version: 1.0.0
----------------------------------------------------------------------------------------
*/

log.info "[DrugSol] Nextflow ${nextflow.version} | DSL2 enabled"

// ============================================================================
// PARAMETERS (all defaults defined in nextflow.config)
// ============================================================================

// Ensure outdir is set based on mode if not overridden
if (!params.containsKey('outdir') || params.outdir == null) {
    params.outdir = "${baseDir}/results/${params.mode ?: 'research'}"
}

// ============================================================================
// INCLUDES
// ============================================================================

include { research }  from "${baseDir}/subworkflows/modes/research/research.nf"
include { execution } from "${baseDir}/subworkflows/modes/execution/execution.nf"

// ============================================================================
// HELP MESSAGE
// ============================================================================

def helpMessage() {
    log.info """
    ==========================================================
      D R U G S O L   P I P E L I N E
    ==========================================================
    
    Usage:
        nextflow run main.nf --mode <mode> [options]
    
    Modes:
        research    Train models with cross-validation (default)
        execution   Run inference on new molecules
    
    General Options:
        --outdir        Output directory (default: results/<mode>)
        --help          Show this help message
    
    Research Mode Options:
        --n_iterations  Number of CV iterations (default: 10)
        --n_cv_folds    Number of CV folds (default: 5)
        --random_seed   Random seed (default: 42)
    
    Execution Mode Options:
        --input         Input file with molecules (required)
        --model         Path to model directory (auto-detect if not set)
        --prediction_ph Target pH for solubility (default: 7.4)
    
    Profiles:
        -profile standard   CPU-only execution
        -profile gpu_small  Consumer GPU (e.g., RTX 3070)
        -profile gpu_high   High-end GPU (e.g., A5000, A6000)
    
    Examples:
        # Research mode with 5 CV iterations
        nextflow run main.nf --mode research --n_iterations 5 -profile gpu_small
        
        # Execution mode on new molecules
        nextflow run main.nf --mode execution --input molecules.csv -profile standard
    
    ==========================================================
    """.stripIndent()
}

// ============================================================================
// HEADER
// ============================================================================

def printHeader() {
    log.info """
    ==========================================================
      D R U G S O L   P I P E L I N E
    ==========================================================
    Mode                : ${params.mode}
    Input Data          : ${params.input ?: 'N/A'}
    Output Dir          : ${params.outdir}
    ----------------------------------------------------------
    [Execution Params]
    Model Override      : ${params.model ?: 'Auto-detect from results'}
    Temp Range (K)      : ${params.temperature_K_Min ?: '-'} to ${params.temperature_K_Max ?: '-'}
    ==========================================================
    """.stripIndent()
}

// ============================================================================
// VALIDATION
// ============================================================================

def validateExecutionEnvironment() {
    if (params.model) return
    
    def product_dir = file("${baseDir}/results/research/final_product/drugsol_model")
    def model_card  = file("${product_dir}/model_card.json")
    
    boolean models_exist = product_dir.isDirectory() && model_card.exists()
    
    if (!models_exist) {
        error """
        [ERROR] Execution mode requires a trained model.
        
        Reason:
        1. No '--model' parameter was provided.
        2. A valid 'Final Product' was not found.
           
           Checked Location: 
           - ${product_dir}
           
           Missing Artifact:
           - model_card.json 
        
        Solution:
        - Run with '--mode research' first to generate the final product.
        - OR provide a specific model path with '--model /path/to/model'.
        - OR ensure '--outdir' matches the directory used in the research phase.
        """
    }
}

// ============================================================================
// MAIN WORKFLOW
// ============================================================================

workflow drugsol {
    
    // Show help if requested
    if (params.help) {
        helpMessage()
        exit 0
    }
    
    printHeader()
    
    // Pass all parameters to subworkflows
    def run_params = Channel.value(params)
    
    if (params.mode == 'research') {
        
        research(run_params)
        
    } else if (params.mode == 'execution') {
        
        validateExecutionEnvironment()
        execution(run_params)
        
    } else {
        error "[ERROR] Invalid mode: '${params.mode}'. Valid options: 'research', 'execution'."
    }
}

// ============================================================================
// DEFAULT WORKFLOW
// ============================================================================

workflow {
    drugsol()
}
