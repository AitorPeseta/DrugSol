nextflow.enable.dsl = 2

/*
 * DrugSol - Main Workflow Entrypoint
 * Structure: Routing based on --mode parameter (research vs. execution)
 */

log.info "[DrugSol] Nextflow ${nextflow.version} | DSL2 enabled"

/************************************
 * PARAMS
 ************************************/

// --- General Parameters ---
params.mode      = params.mode      ?: 'research'           // Options: 'research', 'execution'
params.outdir    = params.outdir    ?: "${baseDir}/results/${params.mode}" // Output directory
params.input     = params.input     ?: null                 // Path to unified table (CSV/TSV/Parquet)

// --- Research Specific ---
params.stratify  = params.stratify  ?: 'solvent,T_bin'      // Columns for stratified splitting

// --- Execution (Inference) Specific ---
params.model             = params.model             ?: null // Specific .pkl artifact (optional override)
params.solvent           = params.solvent           ?: null
params.temperature_K_Min = params.temperature_K_Min ?: null
params.temperature_K_Max = params.temperature_K_Max ?: null

/************************************
 * INCLUDES
 ************************************/
include { research }  from "${baseDir}/subworkflows/modes/research/research.nf"
include { execution } from "${baseDir}/subworkflows/modes/execution/execution.nf"

/************************************
 * HELPERS & VALIDATION
 ************************************/

def print_header() {
    log.info """\
    ==========================================================
      D R U G S O L   P I P E L I N E
    ==========================================================
    Mode                : ${params.mode}
    Input Data          : ${params.input ?: 'N/A'}
    Output Dir          : ${params.outdir}
    ----------------------------------------------------------
    [Execution Params]
    Model Override      : ${params.model ?: 'Auto-detect from results'}
    Solvent             : ${params.solvent ?: 'N/A'}
    Temp Range (K)      : ${params.temperature_K_Min ?: '-'} to ${params.temperature_K_Max ?: '-'}
    ==========================================================
    """.stripIndent()
}

// Function to validate execution requirements
def validate_execution_environment() {
    if (params.model) return
    def product_dir = file("${baseDir}/results/research/final_product/drugsol_model")
    def model_card  = file("${product_dir}/model_card.json")

    boolean models_exist = product_dir.isDirectory() && model_card.exists()

    if (!models_exist) {
        error """
        [ERROR] Execution mode requires a trained model.
        
        Reason:
        1. No '--model' parameter was provided.
        2. A valid 'Final Product' was not found in the output directory.
           
           Checked Location: 
           - ${product_dir}
           
           Missing Artifact:
           - model_card.json 
           (This file validates that the build process finished successfully, 
            regardless of whether the strategy was stack, blend, or single).
        
        Solution:
        - Run with '--mode research' first to generate the final product.
        - OR provide a specific model path with '--model /path/to/model'.
        - OR ensure '--outdir' matches the directory used in the research phase.
        """
    }
}

/************************************
 * MAIN WORKFLOW
 ************************************/

workflow drugsol {
    
    print_header()
    
    // Pass all parameters as a single map to subworkflows
    def run_params = Channel.value(params)

    if (params.mode == 'research') {
        
        research(run_params)

    } else if (params.mode == 'execution') {
        
        // Validate before running
        validate_execution_environment()
        
        execution(run_params)

    } else {
        error "[ERROR] Invalid mode: '${params.mode}'. Valid options are: 'research', 'execution'."
    }
}