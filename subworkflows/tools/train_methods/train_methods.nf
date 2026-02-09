#!/usr/bin/env nextflow
/*
========================================================================================
    Train Methods Subworkflow: Model Training and Evaluation Pipeline
========================================================================================
    Multi-level machine learning pipeline implementing stacking/blending ensemble:
    
    Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Level 0 (Base Models)                                                          │
    │  ├── XGBoost      (GBM)     ─┐                                                  │
    │  ├── LightGBM     (GBM)     ─┼──► Mordred descriptors                           │
    │  ├── CatBoost     (GBM)     ─┘                                                  │
    │  ├── Chemprop     (GNN)     ────► SMILES + molecular graphs                     │
    │  └── Physics      (Baseline)────► Temperature + physicochemical properties      │
    │                                                                                 │
    │  Level 1 (Meta-Learners)                                                        │
    │  ├── Blending     ──► Weighted average of Level 0 predictions                   │
    │  └── Stacking     ──► Ridge regression on Level 0 OOF predictions               │
    └─────────────────────────────────────────────────────────────────────────────────┘
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

// Cross-Validation Setup
include { make_folds }           from '../../../modules/make_folds/make_folds.nf'

// Out-of-Fold Training
include { train_oof_gbm }        from '../../../modules/train_oof_gbm/train_oof_gbm.nf'
include { train_oof_chemprop }   from '../../../modules/train_oof_chemprop/train_oof_chemprop.nf'
include { train_oof_physics }    from '../../../modules/train_oof_physics/train_oof_physics.nf'

// Meta-Learning
include { meta_stack_blend }     from '../../../modules/meta_stack_blend/meta_stack_blend.nf'

// Hyperparameter Consolidation
include { consolidate_params as consolidate_params_gbm } from '../../../modules/consolidate_params/consolidate_params.nf'
include { consolidate_params as consolidate_params_gnn } from '../../../modules/consolidate_params/consolidate_params.nf'

// Full Training
include { train_full_gbm }       from '../../../modules/train_full_gbm/train_full_gbm.nf'
include { train_full_chemprop }  from '../../../modules/train_full_chemprop/train_full_chemprop.nf'
include { train_full_physics }   from '../../../modules/train_full_physics/train_full_physics.nf'

// Inference and Evaluation
include { final_infer_master }                        from '../../../modules/final_infer_master/final_infer_master.nf'
include { cross_validation }                          from '../../../modules/cross_validation/cross_validation.nf'
include { final_report }                              from '../../../modules/final_report/final_report.nf'
include { final_report as final_report_cross }        from '../../../modules/final_report/final_report.nf'

/*
========================================================================================
    SUBWORKFLOW: TRAIN_METHODS
========================================================================================
*/

workflow train_methods {

    take:
        FINAL_TRAIN         // [id, file] Training data with Mordred descriptors
        FINAL_TEST          // [id, file] Test data with Mordred descriptors
        FINAL_TRAIN_SMILE   // [id, file] Training data with RDKit features + SMILES
        FINAL_TEST_SMILE    // [id, file] Test data with RDKit features + SMILES
        OUTDIR_VAL          // Output directory path

    main:
        
        // =====================================================================
        // Script Definitions
        // =====================================================================
        script_folds       = Channel.value( file("${baseDir}/bin/make_folds.py") )
        script_oof_gbm     = Channel.value( file("${baseDir}/bin/train_oof_gbm.py") )
        script_oof_gnn     = Channel.value( file("${baseDir}/bin/train_oof_chemprop.py") )
        script_oof_physics = Channel.value( file("${baseDir}/bin/train_oof_physics.py") )
        script_consol      = Channel.value( file("${baseDir}/bin/consolidate_params.py") )
        script_stack       = Channel.value( file("${baseDir}/bin/meta_stack_blend.py") )
        script_full_gbm    = Channel.value( file("${baseDir}/bin/train_full_gbm.py") )
        script_full_gnn    = Channel.value( file("${baseDir}/bin/train_full_chemprop.py") )
        script_full_physics= Channel.value( file("${baseDir}/bin/train_full_physics.py") )
        script_infer       = Channel.value( file("${baseDir}/bin/final_infer_master.py") )
        script_cross       = Channel.value( file("${baseDir}/bin/cross_validation.py") )
        script_report      = Channel.value( file("${baseDir}/bin/final_report.py") )

        // =====================================================================
        // STAGE 1: Cross-Validation Fold Generation
        // =====================================================================
        make_folds( FINAL_TRAIN_SMILE, OUTDIR_VAL, script_folds )
        ch_folds = make_folds.out  // [id, folds.parquet]

        // =====================================================================
        // STAGE 2: Out-of-Fold (OOF) Training - Level 0 Base Models
        // =====================================================================

        // GBM (XGBoost + LightGBM + CatBoost)
        input_gbm = FINAL_TRAIN.join(ch_folds) 
        train_oof_gbm( input_gbm, OUTDIR_VAL, script_oof_gbm )
        
        ch_oof_xgb  = train_oof_gbm.out.OOF_XGB
        ch_oof_lgbm = train_oof_gbm.out.OOF_LGBM
        ch_oof_cat  = train_oof_gbm.out.OOF_CAT
        HP_DIR      = train_oof_gbm.out.HP_DIR

        // Chemprop (GNN)
        input_gnn = FINAL_TRAIN_SMILE.join(ch_folds)
        train_oof_chemprop( input_gnn, OUTDIR_VAL, script_oof_gnn )
        
        ch_oof_gnn = train_oof_chemprop.out.OOF_GNN
        BEST_GNN   = train_oof_chemprop.out.BEST_PARAMS

        // Physics Baseline
        input_physics = FINAL_TRAIN_SMILE.join(ch_folds)
        train_oof_physics( input_physics, OUTDIR_VAL, script_oof_physics )
        ch_oof_physics = train_oof_physics.out.OOF_PHYSICS

        // =====================================================================
        // Hyperparameter Consolidation
        // =====================================================================
        ch_list_hp_gbm = HP_DIR.map{ it[1] }.collect()
        ch_list_hp_gnn = BEST_GNN.map{ it[1] }.collect()
        
        consolidate_params_gbm(ch_list_hp_gbm, script_consol)
        consolidate_params_gnn(ch_list_hp_gnn, script_consol)

        // =====================================================================
        // STAGE 3: Meta-Learning (Stacking/Blending) - Level 1
        // =====================================================================
        ch_meta_input = ch_oof_lgbm
            .join(ch_oof_xgb)
            .join(ch_oof_cat)
            .join(ch_oof_gnn)
            .join(ch_oof_physics)

        meta_stack_blend(ch_meta_input, OUTDIR_VAL, script_stack)
        
        ch_weights_json = meta_stack_blend.out.BLEND_WEIGHTS
        ch_stack_model  = meta_stack_blend.out.STACK_MODEL
        ch_oof_combined = meta_stack_blend.out.OOF_COMBINED
        ch_all_oof = ch_oof_combined.map { id, file -> file }.collect()

        // =====================================================================
        // STAGE 4: Full Model Training (Production Artifacts)
        // =====================================================================

        // Full GBM Training
        input_full_gbm = FINAL_TRAIN.join(HP_DIR)
        train_full_gbm( input_full_gbm, OUTDIR_VAL, script_full_gbm )
        ch_models_gbm_dir = train_full_gbm.out.MODELS_DIR

        // Full Chemprop Training
        input_full_gnn = FINAL_TRAIN_SMILE.join(BEST_GNN)
        train_full_chemprop(input_full_gnn, OUTDIR_VAL, script_full_gnn)
        ch_models_gnn_dir = train_full_chemprop.out.CHEMPROP_DIR

        // Full Physics Baseline Training
        train_full_physics(FINAL_TRAIN_SMILE, OUTDIR_VAL, script_full_physics)
        ch_model_physics_json = train_full_physics.out.MODEL_JSON

        // =====================================================================
        // STAGE 5: Final Inference (Test Set Evaluation)
        // =====================================================================
        ch_infer_input = FINAL_TEST
            .join(FINAL_TEST_SMILE)
            .join(ch_models_gbm_dir)
            .join(ch_models_gnn_dir)
            .join(ch_model_physics_json)
            .join(ch_weights_json)
            .join(ch_stack_model)
        
        final_infer_master(
            ch_infer_input,
            OUTDIR_VAL,
            script_infer
        )
        
        level0  = final_infer_master.out.LEVEL0
        blend   = final_infer_master.out.BLEND
        stack   = final_infer_master.out.STACK
        metrics = final_infer_master.out.METRICS

        // =====================================================================
        // STAGE 6: Cross-Validation Metrics Aggregation
        // =====================================================================
        ch_all_level0  = level0.map { it[1] }.collect()
        ch_all_blend   = blend.map  { it[1] }.collect()
        ch_all_stack   = stack.map  { it[1] }.collect()
        ch_all_metrics = metrics.map { it[1] }.collect()
        
        cross_validation(
            ch_all_level0,
            ch_all_blend,
            ch_all_stack,
            ch_all_metrics,
            script_cross
        )

        level0_cross_val = cross_validation.out.LEVEL0
        blend_cross_val  = cross_validation.out.BLEND
        stack_cross_val  = cross_validation.out.STACK

        // =====================================================================
        // STAGE 7: Report Generation
        // =====================================================================

        // Per-Iteration Reports
        ch_report_input = level0
            .join(blend)
            .join(stack)
            .join(FINAL_TEST_SMILE)
            .join(ch_oof_combined)
            
        final_report(
            ch_report_input,
            OUTDIR_VAL,
            script_report,
            false
        )

        // Cross-Validation Summary Report
        ch_report_input_cross_validation = level0_cross_val
            .join(blend_cross_val)
            .join(stack_cross_val) 
            .map { id, l0, bl, st ->
                tuple(id, l0, bl, st, l0, l0)
            }
            
        final_report_cross(
            ch_report_input_cross_validation,
            OUTDIR_VAL,
            script_report,
            true
        )

    emit:
        BEST_STRATEGY  = cross_validation.out.BEST_STRATEGY
        OOF_ALL_CONCAT = ch_all_oof
        BEST_HP_GBM    = consolidate_params_gbm.out.BEST_PARAMS
        BEST_HP_GNN    = consolidate_params_gnn.out.BEST_PARAMS
}

/*
========================================================================================
    THE END
========================================================================================
*/
