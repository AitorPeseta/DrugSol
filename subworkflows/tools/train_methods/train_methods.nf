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

// --- Full Training (Production Artifacts) ---
include { train_full_gbm }       from '../../../modules/train_full_gbm/train_full_gbm.nf'
include { train_full_chemprop }  from '../../../modules/train_full_chemprop/train_full_chemprop.nf'
include { train_full_tpsa }      from '../../../modules/train_full_tpsa/train_full_tpsa.nf'

// --- Inference ---
include { final_infer_master }   from '../../../modules/final_infer_master/final_infer_master.nf'

/**
 * WORKFLOW: train_methods
 * -----------------------
 * 1. Generate CV Folds.
 * 2. Train OOF Models (XGB, LGBM, Chemprop, TPSA) to find best Hyperparams.
 * 3. Learn Blending Weights (Stacking) from OOF predictions.
 * 4. Retrain "Full" models on 100% of training data.
 * 5. Run Inference on Test Set using the Full Models + Weights.
 */

workflow train_methods {

    take:
        ch_train_gbm      // Parquet with Mordred features (for XGB/LGBM)
        ch_test_gbm       // Parquet with Mordred features
        ch_train_smiles   // Parquet with SMILES (for Chemprop/TPSA)
        ch_test_smiles    // Parquet with SMILES
        outdir_val        // Output directory

    main:

        // --- Scripts Definition ---
        def script_folds      = file("${baseDir}/bin/make_folds.py")
        def script_oof_gbm    = file("${baseDir}/bin/train_oof_gbm.py")
        def script_oof_gnn    = file("${baseDir}/bin/train_oof_chemprop.py")
        def script_oof_tpsa   = file("${baseDir}/bin/train_oof_tpsa.py")
        def script_stack      = file("${baseDir}/bin/meta_stack_blend.py")
        
        def script_full_gnn   = file("${baseDir}/bin/train_full_chemprop.py")
        def script_full_tpsa  = file("${baseDir}/bin/train_full_tpsa.py")
        def script_full_gbm   = file("${baseDir}/bin/train_full_gbm.py")
        def script_infer      = file("${baseDir}/bin/final_infer_master.py")


        // ============================================================
        // 1. CROSS-VALIDATION FOLDS
        // ============================================================
        // We generate folds once so all models use the exact same splits
        make_folds(ch_train_smiles, outdir_val, script_folds)
        def ch_folds = make_folds.out


        // ============================================================
        // 2. OOF TRAINING (Hyperparam Search & Validation)
        // ============================================================
        
        // A. GBMs (LightGBM + XGBoost)
        train_oof_gbm(ch_train_gbm, outdir_val, script_oof_gbm, ch_folds)
        def ch_oof_lgbm   = train_oof_gbm.out.OFF_LGBM
        def ch_oof_xgb    = train_oof_gbm.out.OFF_XGB
        def ch_hp_dir     = train_oof_gbm.out.HP_DIR // Best params found

        // B. Graph Neural Network (Chemprop)
        train_oof_chemprop(ch_train_smiles, outdir_val, script_oof_gnn, ch_folds)
        def ch_oof_gnn    = train_oof_chemprop.out.OFF_GNN
        def ch_best_gnn   = train_oof_chemprop.out.BEST_GNN // Best checkpoint/config

        // C. Baseline (TPSA Regression)
        train_oof_tpsa(ch_train_smiles, outdir_val, script_oof_tpsa, ch_folds)
        def ch_oof_tpsa   = train_oof_tpsa.out.OFF_TPSA


        // ============================================================
        // 3. META-LEARNING (Stacking/Blending)
        // ============================================================
        
        // Combine all OOF predictions into a single tuple for the blender
        def ch_meta_input = ch_oof_lgbm
            .combine(ch_oof_xgb)
            .combine(ch_oof_gnn)
            .combine(ch_oof_tpsa)
            .map { lgbm, xgb, gnn, tpsa -> tuple(lgbm, xgb, gnn, tpsa) }

        meta_stack_blend(ch_meta_input, outdir_val, script_stack, "_v1")
        
        // Capture weights file. 
        def ch_weights_json = meta_stack_blend.out.BLEND_WEIGHTS
        def ch_stack_model  = meta_stack_blend.out.STACK_MODEL


        // ============================================================
        // 4. FULL TRAINING (Retrain on 100% Data)
        // ============================================================

        // A. GBMs (Using best Hyperparams from OOF)
        train_full_gbm(ch_train_gbm, outdir_val, script_full_gbm, ch_hp_dir)
        def ch_models_gbm_dir = train_full_gbm.out.MODELS_DIR

        // B. Chemprop (Using best Config from OOF)
        train_full_chemprop(ch_train_smiles, outdir_val, script_full_gnn, ch_best_gnn)
        def ch_models_gnn_dir = train_full_chemprop.out.CHEMPROP_DIR

        // C. TPSA
        train_full_tpsa(ch_train_smiles, outdir_val, script_full_tpsa)
        def ch_model_tpsa_pkl = train_full_tpsa.out.TPSA_MODEL


        // ============================================================
        // 5. FINAL INFERENCE (Test Set)
        // ============================================================
        
        final_infer_master(
            ch_test_gbm,        // Input Features (Mordred)
            ch_test_smiles,     // Input SMILES
            ch_models_gbm_dir,  // Folder containing xgb.pkl and lgbm.pkl
            ch_models_gnn_dir,  // Folder containing Chemprop checkpoints
            ch_model_tpsa_pkl,  // TPSA model file
            outdir_val,
            script_infer,
            ch_weights_json,    // Ensemble weights
            ch_stack_model      // (Optional) Stacking meta-model
        )
}