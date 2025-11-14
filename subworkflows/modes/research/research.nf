nextflow.enable.dsl = 2

// --- SOLO includes de subworkflows externos ---
include { common_ingest as COMMON_INGEST } from '../../tools/common_ingest/common_ingest.nf'
include { curate        as CURATE      }   from '../../tools/curate/curate.nf'
include { analysis      as ANALYSIS    }   from '../../tools/analysis/analysis.nf'
include { prepare_data  as PREPARE     }   from '../../tools/prepare_data/prepare_data.nf'
include { train_methods as TRAIN_STACK         }   from '../../tools/train_methods/train_methods.nf'

workflow research {
  take:
    cfg_ch

  main:
    // 1) Ingest
    def OUTDIR_VAL = Channel.value("${params.outdir}/research")
    COMMON_INGEST( OUTDIR_VAL )
    def UNIFIED = COMMON_INGEST.out.unify
    def CHEMBL  = COMMON_INGEST.out.chembl

    // 2) Curate
    CURATE( UNIFIED, OUTDIR_VAL )
    def FINAL_CURATED = CURATE.out.output

    // 3) Preprocesamiento de los datos antes del entrenamiento
    PREPARE( FINAL_CURATED, OUTDIR_VAL, CHEMBL )
    def FINAL_TRAIN       = PREPARE.out.train
    def FINAL_TEST        = PREPARE.out.test
    def FINAL_TRAIN_SMILE = PREPARE.out.train_smiles
    def FINAL_TEST_SMILE = PREPARE.out.test_smiles

    // 4) Análisis de los datos
    ANALYSIS( FINAL_TRAIN, FINAL_TEST, OUTDIR_VAL, FINAL_CURATED )

    // 5) Machine Learning (desde tools/machine_learning/machine_learning.nf)
    TRAIN_STACK( FINAL_TRAIN, FINAL_TEST, FINAL_TRAIN_SMILE, FINAL_TEST_SMILE, OUTDIR_VAL )
}
