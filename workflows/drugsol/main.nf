nextflow.enable.dsl = 2

/*
 * DrugSol - Main entrypoint (módulo)
 * Estructura tipo nf-core: este workflow enruta según --mode
 *
 * NOTA: este archivo define un workflow NOMBRADO `drugsol`
 * para poder ser incluido desde el main raíz.
 */

println "[DrugSol] Nextflow ${nextflow.version} | DSL2 enabled"

/************************************
 *              PARAMS
 ************************************/
params.mode          = params.mode ?: 'research'          // 'research' | 'execution'
params.input         = params.input ?: null               // ruta a tabla unificada (CSV/TSV/Parquet)
params.model         = params.model ?: null               // artifact .pkl (solo para execution)
params.outdir        = params.outdir ?: "${baseDir}/results"   // carpeta de salida (en raíz del repo)
params.stratify      = params.stratify ?: 'solvent,T_bin' // lista csv
params.exp           = params.exp ?: null                 // experimento en execution: 't-sweep' | 'ph-sweep' | ...

// argumentos típicos de execution (opcionales; los recogerá el subworkflow)
params.smiles        = params.smiles ?: null
params.solvent       = params.solvent ?: null
params.temperature_K_Min = params.temperature_K_Min ?: null
params.temperature_K_Max = params.temperature_K_Max ?: null

/************************************
 *           INCLUDES
 *  Usa rutas robustas con baseDir (raíz del repo)
 ************************************/
include { research }   from "${baseDir}/subworkflows/modes/research/research.nf"
include { execution }  from "${baseDir}/subworkflows/modes/execution/execution.nf"

/************************************
 *        HELPERS & VALIDATION
 ************************************/

def _print_header() {
    log.info """\
    ================== DrugSol Pipeline ==================
    mode                : ${params.mode}
    input               : ${params.input ?: '-'}
    model (exec)        : ${params.model ?: '-'}
    outdir              : ${params.outdir}
    stratify            : ${params.stratify}
    exp (execution)     : ${params.exp ?: '-'}
    smiles              : ${params.smiles ?: '-'}
    solvent             : ${params.solvent ?: '-'}
    temperature_K_Min   : ${params.temperature_K_Min ?: '-'}
    temperature_K_Max   : ${params.temperature_K_Max ?: '-'}
    grid (execution)    : ${params.grid ?: '-'}
    ======================================================
    """.stripIndent()
}

/************************************
 *            MAIN WORKFLOW (nombrado)
 ************************************/
workflow drugsol {

    _print_header()
    
    // Despacho según modo
    if (params.mode == 'research') {
        research(
            Channel.value(params)  // pasar todos los params como un único mapa
        )
    } else if (params.mode == 'execution') {
        
        def gnn = "${baseDir}/results/research/training/models_GNN"
        def gbm = "${baseDir}/results/research/training/models_GBM"

        // Solo verificamos la existencia de estos directorios si estamos en 'execution'
        def hasPKLGNN = (params.mode == 'execution' && gnn &&
                        java.nio.file.Files.isDirectory(
                            (gnn instanceof java.nio.file.Path) ? gnn
                                                                    : java.nio.file.Paths.get(gnn.toString().trim())
                        )
                        )
        def hasPKLGBM = (params.mode == 'execution' && gbm &&
                        java.nio.file.Files.isDirectory(
                            (gbm instanceof java.nio.file.Path) ? gbm
                                                                    : java.nio.file.Paths.get(gbm.toString().trim())
                        )
                        )

        if (!(params.model || (hasPKLGNN && hasPKLGBM))) {
            if (params.mode == 'execution') {
                _die("Para --mode execution debes proporcionar un modelo con --model o haber ejecutado antes --research.")
            } else {
                println "[INFO] Modo research: No se necesita modelo, continuando con investigación."
            }
        }else  execution(
                    Channel.value(params)
                )
    }
}
