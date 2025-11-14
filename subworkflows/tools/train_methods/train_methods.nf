nextflow.enable.dsl = 2

// === Includes de módulos ===
include { make_folds } from '../../../modules/make_folds/make_folds.nf'
include { train_oof_gbm } from '../../../modules/train_oof_gbm/train_oof_gbm.nf'
include { train_oof_chemprop } from '../../../modules/train_oof_chemprop/train_oof_chemprop.nf'
include { train_oof_tpsa } from '../../../modules/train_oof_tpsa/train_oof_tpsa.nf'
include { meta_stack_blend } from '../../../modules/meta_stack_blend/meta_stack_blend.nf'
include { train_full_gbm } from '../../../modules/train_full_gbm/train_full_gbm.nf'
include { train_full_chemprop } from '../../../modules/train_full_chemprop/train_full_chemprop.nf'
include { train_full_tpsa } from '../../../modules/train_full_tpsa/train_full_tpsa.nf'
include { final_infer_master } from '../../../modules/final_infer_master/final_infer_master.nf'

/**
 * Workflow: train_methods
 * Entrena y evalúa modelos de ML (stacking/blending) sobre features ya preparadas.
 */
workflow train_methods {

  take:
    FINAL_TRAIN
    FINAL_TEST
    FINAL_TRAIN_SMILE
    FINAL_TEST_SMILE
    OUTDIR_VAL

  main:

    def FOL_PY = Channel.value( file("${baseDir}/bin/make_folds.py") )
    make_folds( FINAL_TRAIN_SMILE, OUTDIR_VAL, FOL_PY )
    def FOL_MERGE = make_folds.out

    def TOG_PY = Channel.value( file("${baseDir}/bin/train_oof_gbm.py") )
    train_oof_gbm( FINAL_TRAIN, OUTDIR_VAL, TOG_PY, FOL_MERGE )
    def OOF_LGBM   = train_oof_gbm.out.OFF_LGBM
    def OOF_XGB    = train_oof_gbm.out.OFF_XGB
    def METRICS_CH = train_oof_gbm.out.METRICS_CH
    def HP_DIR     = train_oof_gbm.out.HP_DIR

    def TOC_PY = Channel.value( file("${baseDir}/bin/train_oof_chemprop.py") )
    train_oof_chemprop( FINAL_TRAIN_SMILE, OUTDIR_VAL, TOC_PY, FOL_MERGE )
    def OOF_GNN  = train_oof_chemprop.out.OFF_GNN
    def BEST_GNN = train_oof_chemprop.out.MANI_GNN

    def TOT_PY = Channel.value( file("${baseDir}/bin/train_oof_tpsa.py") )
    train_oof_tpsa( FINAL_TRAIN_SMILE, OUTDIR_VAL, TOT_PY, FOL_MERGE )
    def OOF_TPSA  = train_oof_tpsa.out.OFF_TPSA
    def BEST_TPSA = train_oof_tpsa.out.BEST_TPSA

    def MSB_PY = Channel.value( file("${baseDir}/bin/meta_stack_blend.py") )
    def META_OOF = OOF_LGBM
                .combine(OOF_XGB)
                .combine(OOF_GNN)
                .combine(OOF_TPSA)
                .map { lgbm, xgb, gnn, tpsa -> tuple(lgbm, xgb, gnn, tpsa) }
    meta_stack_blend( META_OOF, OUTDIR_VAL, MSB_PY, "_v1" )
    def WEIGHTS_JSON = meta_stack_blend.out.BLEND_WEIGHTS        // path channel
                      .map { it.toString() }
                      .ifEmpty { Channel.value(null) }       // garantiza un valor

    def STACK_MODEL_CH = meta_stack_blend.out.STACK_MODEL        // path channel
                      .map { it.toString() }
                      .ifEmpty { Channel.value(null) }


    def TFGNN_PY = Channel.value( file("${baseDir}/bin/train_full_chemprop.py") )
    train_full_chemprop( FINAL_TRAIN_SMILE, OUTDIR_VAL, TFGNN_PY, BEST_GNN )
    def CHEMPROP_DIR = train_full_chemprop.out.CHEMPROP_DIR

    def TFTPSA_PY = Channel.value( file("${baseDir}/bin/train_full_tpsa.py") )
    train_full_tpsa( FINAL_TRAIN_SMILE, OUTDIR_VAL, TFTPSA_PY )
    def TPSA_MODEL = train_full_tpsa.out.TPSA_MODEL

    def TFGBM_PY = Channel.value( file("${baseDir}/bin/train_full_gbm.py") )
    train_full_gbm( FINAL_TRAIN, OUTDIR_VAL, TFGBM_PY, HP_DIR )
    def MODELS_DIR = train_full_gbm.out.MODELS_DIR           
    
    def FIM_PY = Channel.value( file("${baseDir}/bin/final_infer_master.py") )
    final_infer_master(
      FINAL_TEST,       // path
      FINAL_TEST_SMILE, // path
      MODELS_DIR,       // path (dir con xgb.pkl y lgbm.pkl)
      CHEMPROP_DIR,     // path (dir de checkpoints chemprop)
      TPSA_MODEL,
      OUTDIR_VAL,       // val
      FIM_PY,           // path 
      WEIGHTS_JSON,     // val (string o null)
      STACK_MODEL_CH
    )
}
