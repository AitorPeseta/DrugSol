#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
========================================================================================
    DrugSol - Main Workflow Entrypoint
========================================================================================
*/

log.info "[DrugSol] Nextflow ${nextflow.version} | DSL2 enabled"

// ============================================================================
// PARAMETERS
// ============================================================================

if (!params.containsKey('outdir') || params.outdir == null) {
    params.outdir = "${baseDir}/results/${params.mode ?: 'research'}"
}

// Default for skip_env_setup to avoid warning
params.skip_env_setup = params.skip_env_setup ?: false

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
        --outdir          Output directory (default: results/<mode>)
        --help            Show this help message
        --skip_env_setup  Skip environment setup (default: false)
    
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
    Mode         : ${params.mode}
    Output Dir   : ${params.outdir}
    Iterations   : ${params.n_iterations ?: 'N/A'}
    ==========================================================
    """.stripIndent()
}

// ============================================================================
// ENVIRONMENT SETUP
// ============================================================================

def setupEnvironments() {
    if (params.skip_env_setup) {
        log.info "[DrugSol] Skipping environment setup (--skip_env_setup=true)"
        return true
    }
    
    def setup_script = file("${baseDir}/envs/setup_environments.sh")
    def envs_dir = "${baseDir}/envs"
    
    if (!setup_script.exists()) {
        log.warn "[DrugSol] Setup script not found: ${setup_script}"
        return true
    }
    
    // Run setup script (output goes to log file, not console)
    def cmd = ["bash", setup_script.toString(), params.mode, envs_dir]
    def proc = cmd.execute()
    
    // Capture output but don't print verbose micromamba output
    def stdout = new StringBuilder()
    def stderr = new StringBuilder()
    proc.consumeProcessOutput(stdout, stderr)
    proc.waitFor()
    
    // Print only our formatted output (lines starting with [DrugSol])
    stdout.toString().eachLine { line ->
        if (line.startsWith("[DrugSol]")) {
            log.info line
        }
    }
    
    if (proc.exitValue() != 0) {
        log.error "[DrugSol] Environment setup failed!"
        stderr.toString().eachLine { line -> log.error line }
        System.exit(1)
    }
    
    return true
}

// ============================================================================
// VALIDATION
// ============================================================================

def validateExecutionEnvironment() {
    if (params.model) return
    
    def product_dir = file("${baseDir}/results/research/final_product/drugsol_model")
    def model_card  = file("${product_dir}/model_card.json")
    
    if (!product_dir.isDirectory() || !model_card.exists()) {
        error """
        [ERROR] Execution mode requires a trained model.
        Run '--mode research' first or provide '--model /path/to/model'.
        """
    }
}

// ============================================================================
// MAIN WORKFLOW
// ============================================================================

workflow drugsol {
    
    if (params.help) {
        helpMessage()
        exit 0
    }
    
    printHeader()
    setupEnvironments()
    
    def run_params = Channel.value(params)
    
    if (params.mode == 'research') {
        research(run_params)
    } else if (params.mode == 'execution') {
        validateExecutionEnvironment()
        execution(run_params)
    } else {
        error "[ERROR] Invalid mode: '${params.mode}'. Use 'research' or 'execution'."
    }
}

// ============================================================================
// DEFAULT WORKFLOW
// ============================================================================

workflow {
    drugsol()
}
