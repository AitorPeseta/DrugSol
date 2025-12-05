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
params.outdir    = params.outdir    ?: "${baseDir}/${params.mode}/results" // Output directory
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
    // 1. If a specific model is provided, we are good.
    if (params.model) return

    // 2. If no model provided, check if 'research' results exist in the output directory
    def gnn_path = file("${params.outdir}/research/training/models_GNN")
    def gbm_path = file("${params.outdir}/research/training/models_GBM")

    boolean models_exist = gnn_path.isDirectory() && gbm_path.isDirectory()

    if (!models_exist) {
        error """
        [ERROR] Execution mode requires trained models.
        
        Reason:
        1. No '--model' parameter was provided.
        2. Previous research results were not found at:
           - ${gnn_path}
           - ${gbm_path}
        
        Solution:
        - Run with '--mode research' first.
        - OR provide a specific model path with '--model'.
        - OR ensure '--outdir' points to where previous research results are stored.
        """
    } else {
        log.info "[INFO] Models found in ${params.outdir}. Proceeding with execution."
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