nextflow.enable.dsl = 2

// --- SOLO includes de subworkflows externos ---
include { common_ingest as COMMON_INGEST } from '../../tools/common_ingest/common_ingest.nf'
include { curate        as CURATE      }   from '../../tools/curate/curate.nf'
include { analysis      as ANALYSIS    }   from '../../tools/analysis/analysis.nf'
include { prepare_data  as PREPARE     }   from '../../tools/prepare_data/prepare_data.nf'
include { final_infer_master } from '../../../modules/final_infer_master/final_infer_master.nf'

workflow execution {
  take:
    cfg_ch

  main:
    def OUTDIR_VAL = Channel.value("${params.outdir}/execution")

    // 1) Ingest
    COMMON_INGEST( OUTDIR_VAL )
    def UNIFIED = COMMON_INGEST.out.unify
    def CHEMBL  = COMMON_INGEST.out.chembl

    // 2) Curate (desde tools/curate/curate.nf)
    CURATE( UNIFIED, OUTDIR_VAL )
    def FINAL_CURATED = CURATE.out.output

    // 3) Preprocesamiento de los datos antes del entrenamiento
    PREPARE( FINAL_CURATED, OUTDIR_VAL, CHEMBL )
    def FINAL_TEST        = PREPARE.out.test
    def FINAL_TEST_SMILE = PREPARE.out.test_smiles

    // 5) Predicción
    def GNN = "${baseDir}/results/research/training/models_GNN"
    def GBM = "${baseDir}/results/research/training/models_GBM"
    def WEIGHTS_JSON = "${baseDir}/results/research/training/meta_results/blend/weights*"
    def STACK_MODEL_CH = "${baseDir}/results/research/training/meta_results/stack/meta_ridge*"

    def FIM_PY = Channel.value( file("${baseDir}/bin/final_infer_master.py") )
    final_infer_master(
      FINAL_TEST,       // path
      FINAL_TEST_SMILE, // path
      GBM,       // path (dir con xgb.pkl y lgbm.pkl)
      GNN,     // path (dir de checkpoints chemprop)
      OUTDIR_VAL,       // val
      FIM_PY,           // path 
      WEIGHTS_JSON,     // val (string o null)
      STACK_MODEL_CH
    )

}
