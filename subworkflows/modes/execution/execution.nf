nextflow.enable.dsl = 2

// INCLUDES
include { common_ingest as COMMON_INGEST } from '../../tools/common_ingest/common_ingest.nf'
include { curate        as CURATE      }   from '../../tools/curate/curate.nf'
include { prepare_data  as PREPARE     }   from '../../tools/prepare_data/prepare_data.nf'
include { predict_full_pipeline   as PREDICT     }   from '../../../modules/predict_full_pipeline/predict_full_pipeline.nf'

workflow execution {
  take:
    cfg_ch // params pasados desde main

  main:
    def OUTDIR_VAL = Channel.value("${params.outdir}/execution")

    // 1) Ingest (Carga archivo usuario o descarga default)
    COMMON_INGEST( OUTDIR_VAL )
    def UNIFIED = COMMON_INGEST.out.unify

    // 2) Curate (Filtra agua, temperatura, outliers)
    CURATE( UNIFIED, OUTDIR_VAL )
    def FINAL_CURATED = CURATE.out.output

    // 3) Preprocesamiento (Cálculo de features Mordred/RDKit)
    // Pasamos params.iterations/seed aunque no se usen en split de execution
    PREPARE( FINAL_CURATED, OUTDIR_VAL, 1, 42 )
    
    // Obtenemos los datos listos para inferencia
    def FINAL_TEST_GBM   = PREPARE.out.test_gbm    // Mordred features
    def FINAL_TEST_SMILE = PREPARE.out.test_smiles // RDKit features + SMILES

    // 4) Predicción con final_product
    // Localizamos la carpeta final_product. 
    // Asumimos que está en research/results/final_product
    def final_prod_dir = file("${params.outdir}/research/results/final_product")
    
    // Verificación de seguridad
    if (!final_prod_dir.exists()) {
        error "[Execution] No se encuentra final_product en: ${final_prod_dir}. \nEjecuta primero el modo 'research'."
    }

    def script_predict = Channel.value(file("${baseDir}/bin/predict_full_pipeline.py"))

    PREDICT(
        FINAL_TEST_GBM,
        FINAL_TEST_SMILE,
        final_prod_dir,
        script_predict
    )
}