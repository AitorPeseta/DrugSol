nextflow.enable.dsl = 2

// INCLUDES
include { common_ingest as COMMON_INGEST } from '../../tools/common_ingest/common_ingest.nf'
include { curate        as CURATE      }   from '../../tools/curate/curate.nf'
include { prepare_data  as PREPARE     }   from '../../tools/prepare_data/prepare_data.nf'
include { predict_full_pipeline   } from '../../../modules/predict_full_pipeline/predict_full_pipeline.nf'
include { predict_real_solubility } from '../../../modules/predict_real_solubility/predict_real_solubility.nf'


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
    PREPARE( FINAL_CURATED, OUTDIR_VAL, 1, 42, "features_only")
    
    // Obtenemos los datos listos para inferencia
    def FINAL_TEST_GBM   = PREPARE.out.test_gbm.map    { id, file -> file }
    def FINAL_TEST_SMILE = PREPARE.out.test_smiles.map { id, file -> file }

    // 4) Predicción con final_product
    // Localizamos la carpeta final_product. 
    def final_prod_dir = file("${baseDir}/results/research/final_product/drugsol_model")
    
    // Verificación de seguridad
    if (!final_prod_dir.exists()) {
        error "[Execution] No se encuentra final_product en: ${final_prod_dir}. \nEjecuta primero el modo 'research'."
    }

    def script_predict = Channel.value(file("${baseDir}/bin/predict_full_pipeline.py"))

    predict_full_pipeline(
        FINAL_TEST_GBM,
        FINAL_TEST_SMILE,
        final_prod_dir,
        script_predict
    )

    def script_seff = Channel.value(file("${baseDir}/bin/predict_real_solubility.py"))
    
    predict_real_solubility(
        predict_full_pipeline.out,
        OUTDIR_VAL,
        script_seff,
        7.4
    )
}