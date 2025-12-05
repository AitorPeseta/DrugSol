nextflow.enable.dsl = 2

// ============================================================================
// MODULE INCLUDES
// ============================================================================
include { make_folds }           from '../../../modules/make_folds/make_folds.nf'

// --- OOF Training (Research Phase) ---
include { train_oof_gbm }        from '../../../modules/train_oof_gbm/train_oof_gbm.nf'
include { train_oof_chemprop }   from '../../../modules/train_oof_chemprop/train_oof_chemprop.nf'
include { train_oof_tpsa }       from '../../../modules/train_oof_tpsa/train_oof_tpsa.nf'
include { meta_stack_blend }     from '../../../modules/meta_stack_blend/meta_stack_blend.nf'
include { consolidate_params as consolidate_params_gbm }     from '../../../modules/consolidate_params/consolidate_params.nf'
include { consolidate_params as consolidate_params_gnn }     from '../../../modules/consolidate_params/consolidate_params.nf'

// --- Full Training (Production Artifacts) ---
include { train_full_gbm }       from '../../../modules/train_full_gbm/train_full_gbm.nf'
include { train_full_chemprop }  from '../../../modules/train_full_chemprop/train_full_chemprop.nf'
include { train_full_tpsa }      from '../../../modules/train_full_tpsa/train_full_tpsa.nf'

// --- Inference ---
include { final_infer_master }                        from '../../../modules/final_infer_master/final_infer_master.nf'
include { cross_validation }                          from '../../../modules/cross_validation/cross_validation.nf'
include { final_report }                              from '../../../modules/final_report/final_report.nf'
include { final_report as final_report_cross}         from '../../../modules/final_report/final_report.nf'


/**
 * WORKFLOW: train_methods
 * Entrena y evalúa modelos de ML (stacking/blending) sobre features ya preparadas.
 */
workflow train_methods {

  take:
    FINAL_TRAIN         // [id, file] (Mordred)
    FINAL_TEST          // [id, file] (Mordred)
    FINAL_TRAIN_SMILE   // [id, file] (RDKit/SMILES)
    FINAL_TEST_SMILE    // [id, file] (RDKit/SMILES)
    OUTDIR_VAL

  main:
    
    // 0. Scripts Python (Canales de valor)
    def script_folds     = Channel.value( file("${baseDir}/bin/make_folds.py") )
    def script_oof_gbm   = Channel.value( file("${baseDir}/bin/train_oof_gbm.py") )
    def script_oof_gnn   = Channel.value( file("${baseDir}/bin/train_oof_chemprop.py") )
    def script_oof_tpsa  = Channel.value( file("${baseDir}/bin/train_oof_tpsa.py") )
    def script_consol    = Channel.value( file("${baseDir}/bin/consolidate_params.py") )
    def script_stack     = Channel.value( file("${baseDir}/bin/meta_stack_blend.py") )
    def script_full_gbm  = Channel.value( file("${baseDir}/bin/train_full_gbm.py") )
    def script_full_gnn  = Channel.value( file("${baseDir}/bin/train_full_chemprop.py") )
    def script_full_tpsa = Channel.value( file("${baseDir}/bin/train_full_tpsa.py") )
    def script_infer     = Channel.value( file("${baseDir}/bin/final_infer_master.py") )
    def script_cross     = Channel.value( file("${baseDir}/bin/cross_validation.py") )
    def script_report    = Channel.value( file("${baseDir}/bin/final_report.py") )

    // ============================================================
    // 1. MAKE FOLDS (Cross-Validation Setup)
    // ============================================================
    // Usamos el Train de RDKit que es más ligero para hacer los folds (solo necesita logS y scaffold)
    make_folds( FINAL_TRAIN_SMILE, OUTDIR_VAL, script_folds )
    def ch_folds = make_folds.out // [id, folds.parquet]

    // ============================================================
    // 2. OOF TRAINING (Level 0 Models)
    // ============================================================

    // A. GBM (XGBoost + LightGBM) -> Usa features Mordred
    // Unimos Train Mordred con Folds por ID
    def input_gbm = FINAL_TRAIN.join(ch_folds) 
    train_oof_gbm( input_gbm, OUTDIR_VAL, script_oof_gbm )
    
    def ch_oof_xgb  = train_oof_gbm.out.OFF_XGB
    def ch_oof_lgbm = train_oof_gbm.out.OFF_LGBM
    def HP_DIR      = train_oof_gbm.out.HP_DIR

    // B. CHEMPROP (GNN) -> Usa SMILES
    def input_gnn = FINAL_TRAIN_SMILE.join(ch_folds)
    
    train_oof_chemprop(
        input_gnn, 
        OUTDIR_VAL, 
        script_oof_gnn
    )
    
    def ch_oof_gnn  = train_oof_chemprop.out.OFF_GNN
    def BEST_GNN    = train_oof_chemprop.out.BEST_GNN

    // C. TPSA (Baseline) -> Usa SMILES/RDKit features
    def input_tpsa = FINAL_TRAIN_SMILE.join(ch_folds)
    train_oof_tpsa( input_tpsa, OUTDIR_VAL, script_oof_tpsa )
    def ch_oof_tpsa = train_oof_tpsa.out.OFF_TPSA

    
    def ch_list_hp_gbm = HP_DIR.map{ it[1] }.collect()
    def ch_list_hp_gnn = BEST_GNN.map{ it[1] }.collect()
    consolidate_params_gbm(ch_list_hp_gbm, script_consol)
    consolidate_params_gnn(ch_list_hp_gnn, script_consol)

    // ============================================================
    // 3. META-LEARNING (Stacking/Blending)
    // ============================================================
    
    // UNIÓN CRÍTICA: Usamos .join() para asegurar que mezclamos la misma iteración
    def ch_meta_input = ch_oof_lgbm
        .join(ch_oof_xgb)
        .join(ch_oof_gnn)
        .join(ch_oof_tpsa)
        // Resultado: [id, lgbm, xgb, gnn, tpsa]

    meta_stack_blend(ch_meta_input, OUTDIR_VAL, script_stack)
    
    def ch_weights_json = meta_stack_blend.out.BLEND_WEIGHTS
    def ch_stack_model  = meta_stack_blend.out.STACK_MODEL
    def ch_oof_combined = meta_stack_blend.out.OOF_COMBINED
    def ch_all_oof = ch_oof_combined.map { id, file -> file }.collect()

    // ============================================================
    // 4. FULL TRAINING (Production Models)
    // ============================================================
    // Entrenamos con TODO el dataset usando los mejores hiperparámetros

    // A. GBM
    // Necesitamos unir el Train Mordred con los HPs
    def input_full_gbm = FINAL_TRAIN.join(HP_DIR)
    train_full_gbm( input_full_gbm, OUTDIR_VAL, script_full_gbm )
    def ch_models_gbm_dir = train_full_gbm.out.MODELS_DIR

    // B. GNN
    def input_full_gnn = FINAL_TRAIN_SMILE.join(BEST_GNN)

    train_full_chemprop(input_full_gnn, OUTDIR_VAL, script_full_gnn)
    def ch_models_gnn_dir = train_full_chemprop.out.CHEMPROP_DIR

    // C. TPSA
    train_full_tpsa(FINAL_TRAIN_SMILE, OUTDIR_VAL, script_full_tpsa)
    def ch_model_tpsa_pkl = train_full_tpsa.out.TPSA_MODEL


    // ============================================================
    // 5. FINAL INFERENCE (Test Set)
    // ============================================================
    
    // CRUCIAL: Unir todos los inputs por ID para evitar el producto cartesiano
    // Estructura deseada: [id, test_gbm, test_smi, mod_gbm, mod_gnn, mod_tpsa, weights, stack]
    
    def ch_infer_input = FINAL_TEST          // [id, test_gbm]
        .join(FINAL_TEST_SMILE)              // + [test_smi]
        .join(ch_models_gbm_dir)             // + [mod_gbm]
        .join(ch_models_gnn_dir)             // + [mod_gnn]
        .join(ch_model_tpsa_pkl)             // + [mod_tpsa]
        .join(ch_weights_json)               // + [weights]
        .join(ch_stack_model)                // + [stack]
    
    final_infer_master(
        ch_infer_input, // Pasamos la tupla completa
        OUTDIR_VAL,
        script_infer
    )
    
    // Desempaquetar salidas (que ahora llevan ID)
    def level0  = final_infer_master.out.LEVEL0
    def blend   = final_infer_master.out.BLEND
    def stack   = final_infer_master.out.STACK
    def metrics = final_infer_master.out.METRICS

    // ============================================================
    // 6. CROSS-VALIDATION
    // ============================================================

    def ch_all_level0 = level0.map { it[1] }.collect()
    def ch_all_blend  = blend.map  { it[1] }.collect()
    def ch_all_stack  = stack.map  { it[1] }.collect()
    def ch_all_metrics = metrics.map { it[1] }.collect()
    
    cross_validation(
        ch_all_level0,
        ch_all_blend,
        ch_all_stack,
        ch_all_metrics,
        script_cross
    )

    def level0_cross_val = cross_validation.out.LEVEL0
    def blend_cross_val = cross_validation.out.BLEND
    def stack_cross_val = cross_validation.out.STACK

    // ============================================================
    // 7. FINAL REPORT
    // ============================================================

    // Necesitamos unir todos los datos para el reporte
    // Test Level0, Blend, Stack, Metadata (Test Smiles para Temp), OOF (Train)
    
    def ch_report_input = level0
        .join(blend)
        .join(stack)
        .join(FINAL_TEST_SMILE) // Metadata
        .join(ch_oof_combined)  // OOF Predictions
        
    final_report(
        ch_report_input,
        OUTDIR_VAL,
        script_report,
        false
    )

    // Los mismos pasos que el anterior pero con los datos del cross-validation
    def ch_report_input_cross_validation = level0_cross_val
        .join(blend_cross_val)
        .join(stack_cross_val) 
        
        .map { id, l0, bl, st ->
            return tuple(id, l0, bl, st, l0, l0)
        }
        
    final_report_cross(
        ch_report_input_cross_validation,
        OUTDIR_VAL,
        script_report,
        true
    )

    emit:
    BEST_STRATEGY = cross_validation.out.BEST_STRATEGY
    OOF_ALL_CONCAT = ch_all_oof
    BEST_HP_GBM    = consolidate_params_gbm.out.BEST_PARAMS
    BEST_HP_GNN    = consolidate_params_gnn.out.BEST_PARAMS
}