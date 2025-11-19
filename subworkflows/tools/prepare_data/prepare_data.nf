include { stratified_split }                                from '../../../modules/stratified_split/stratified_split.nf'
include { make_features_mordred as make_features_mordred_train } from '../../../modules/make_features_mordred/make_features_mordred.nf'
include { make_features_mordred as make_features_mordred_test  } from '../../../modules/make_features_mordred/make_features_mordred.nf'
include { make_features_rdkit as make_features_rdkit_train } from '../../../modules/make_features_rdkit/make_features_rdkit.nf'
include { make_features_rdkit as make_features_rdkit_test  } from '../../../modules/make_features_rdkit/make_features_rdkit.nf'
include { align_feature_columns as align_feature_columns_mordred}                           from '../../../modules/align_feature_columns/align_feature_columns.nf'
include { align_feature_columns as align_feature_columns_rdkit}                           from '../../../modules/align_feature_columns/align_feature_columns.nf'
include { dropnan_rows as dropnan_rows_train_smile }              from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows as dropnan_rows_test_smile  }              from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows as dropnan_rows_train }              from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows as dropnan_rows_test  }              from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { make_fingerprints }     from '../../../modules/make_fingerprints/make_fingerprints.nf'
include { enrich_ionization_features }     from '../../../modules/enrich_ionization_features/enrich_ionization_features.nf'
include { standardize_smiles }                              from '../../../modules/standardize_smiles/standardize_smiles.nf'
include { merge_drug_flag }                                 from '../../../modules/merge_drug_flag/merge_drug_flag.nf'
include { filter_features as filter_features_train }        from '../../../modules/filter_features/filter_features.nf'
include { filter_features as filter_features_test}          from '../../../modules/filter_features/filter_features.nf'




/**
 * Workflow: prepare_data
 * Toma un dataset unificado, añade flags, limpia, hace split y prepara features train/test alineadas.
 */
workflow prepare_data {

  take:
    FILTER_O    
    OUTDIR_VAL    // val con el outdir
    CHEMBL        // base de referencia para merge_drug_flag

  main: 

    def ES_PY = Channel.value( file("${baseDir}/bin/stratified_split.py") )
    def MFM_PY = Channel.value( file("${baseDir}/bin/make_features_mordred.py") )
    def MFR_PY = Channel.value( file("${baseDir}/bin/make_features_rdkit.py") )
    def AL_PY = Channel.value( file("${baseDir}/bin/align_feature_columns.py") )
    def DN_PY = Channel.value( file("${baseDir}/bin/dropnan_rows.py") )
    def FN_PY = Channel.value( file("${baseDir}/bin/make_fingerprints.py") )
    def EIF_PY = Channel.value( file("${baseDir}/bin/enrich_ionization_features.py") )
    def STD_PY = Channel.value( file("${baseDir}/bin/standardize_smiles.py") )
    def MERG_PY = Channel.value( file("${baseDir}/bin/merge_drug_flag.py") )
    def FF_PY = Channel.value( file("${baseDir}/bin/filter_features.py") )
    def mode = params.mode
    def input_val  = params.input
    def hasInput   = (input_val && input_val.toString().trim() && input_val.toString().trim() != '-')
    def FINAL_TRAIN_GBM   = Channel.empty()
    def FINAL_TEST_GBM    = Channel.empty()
    def FINAL_TRAIN_SMILE = Channel.empty()
    def FINAL_TEST_SMILE  = Channel.empty() 
    
    // --- Estandarizar SMILES / InChIKey ---
    standardize_smiles( FILTER_O, OUTDIR_VAL, STD_PY )
    def CURATED = standardize_smiles.out

    // --- Flag de fármaco (merge con CHEMBL) ---
    merge_drug_flag( CURATED, CHEMBL, OUTDIR_VAL, MERG_PY )
    def MERGED = merge_drug_flag.out

    enrich_ionization_features(MERGED, OUTDIR_VAL, EIF_PY )
    def IONIZATED = enrich_ionization_features.out

    make_fingerprints(IONIZATED, OUTDIR_VAL, FN_PY, "cluster_ecfp4_0p7")
    def FINGER = make_fingerprints.out

    if (mode != 'execution') {
      // --- 1) Split estratificado por grupo ---
      stratified_split( FINGER, OUTDIR_VAL, ES_PY )
      def TRAIN = stratified_split.out.train
      def TEST  = stratified_split.out.test

      // ===========GBM==============

      // --- 2) Features (Mordred) para train y test ---
      make_features_mordred_train( TRAIN, OUTDIR_VAL, MFM_PY, "train" )
      make_features_mordred_test(  TEST,  OUTDIR_VAL, MFM_PY, "test"  )
      def FEAT_TRAIN_MR = make_features_mordred_train.out
      def FEAT_TEST_MR  = make_features_mordred_test.out

      filter_features_train( FEAT_TRAIN_MR, OUTDIR_VAL, FF_PY, "train")
      filter_features_test( FEAT_TEST_MR, OUTDIR_VAL, FF_PY, "test")
      def FEAT_FILTER_TRAIN = filter_features_train.out
      def FEAT_FILTER_TEST = filter_features_test.out

      def feat_train_align_mr = FEAT_FILTER_TRAIN
      def feat_train_drop_mr  = FEAT_FILTER_TRAIN
      def PAIRS_CH = feat_train_align_mr.combine( FEAT_FILTER_TEST )

      // --- 3) Alinear columnas (test respecto a train)
      align_feature_columns_mordred( PAIRS_CH, OUTDIR_VAL, AL_PY )
      def AL_TEST_CH = align_feature_columns_mordred.out

      // --- 4) Drop NaN (train sin alinear, test ya alineado)
      dropnan_rows_train( feat_train_drop_mr, OUTDIR_VAL, DN_PY, "final_train", "train" )
      dropnan_rows_test(  AL_TEST_CH, OUTDIR_VAL, DN_PY, "final_test",  "test"  )

      FINAL_TRAIN_GBM = dropnan_rows_train.out
      FINAL_TEST_GBM  = dropnan_rows_test.out

      // ===========GNN Y TPSA==============

      // --- 2) Features (RDKit) para train y test ---
      make_features_rdkit_train( TRAIN, OUTDIR_VAL, MFR_PY, "train" )
      make_features_rdkit_test(  TEST,  OUTDIR_VAL, MFR_PY, "test"  )
      def FEAT_TRAIN_RD = make_features_rdkit_train.out
      def FEAT_TEST_RD  = make_features_rdkit_test.out

      def feat_train_align_rd = FEAT_TRAIN_RD
      def feat_train_drop_rd  = FEAT_TRAIN_RD
      def PAIRS_RD = feat_train_align_rd.combine( FEAT_TEST_RD )

      // --- 3) Alinear columnas (test respecto a train)
      align_feature_columns_rdkit( PAIRS_RD, OUTDIR_VAL, AL_PY )
      def AL_TEST_RD = align_feature_columns_rdkit.out

      // --- 4) Drop NaN (train sin alinear, test ya alineado)
      dropnan_rows_train_smile( feat_train_drop_rd, OUTDIR_VAL, DN_PY, "final_train_smile", "train" )
      dropnan_rows_test_smile(  AL_TEST_RD, OUTDIR_VAL, DN_PY, "final_test_smile",  "test"  )
      
      FINAL_TRAIN_SMILE = dropnan_rows_train_smile.out
      FINAL_TEST_SMILE  = dropnan_rows_test_smile.out

    } else {

      // Genera features de TEST (solo test en ejecución)
      make_features_mordred_test( FINGER, OUTDIR_VAL, MFM_PY, "test" )
      def FEAT_TEST_MR = make_features_mordred_test.out

      make_features_rdkit_test( FINGER, OUTDIR_VAL, MFR_PY, "test" )
      def FEAT_TEST_RD = make_features_rdkit_test.out   

      // Filtra Mordred (solo test), y alinea contra TRAIN precomputado en resources
      filter_features_test( FEAT_TEST_MR, OUTDIR_VAL, FF_PY, "test" )
      def FEAT_FILTER_TEST = filter_features_test.out

      def feat_train_align_mr = Channel.value( file("${baseDir}/resources/train_mordred_featured.parquet") )
      def PAIRS_MR = feat_train_align_mr.combine( FEAT_FILTER_TEST )

      def feat_train_align_rd = Channel.value( file("${baseDir}/resources/train_rdkit_featured.parquet") )
      def PAIRS_RD = feat_train_align_rd.combine( FEAT_TEST_RD )

      // Alinear columnas (test respecto a train)
      align_feature_columns_mordred( PAIRS_MR, OUTDIR_VAL, AL_PY )
      def AL_TEST_MR = align_feature_columns_mordred.out   

      align_feature_columns_rdkit( PAIRS_RD, OUTDIR_VAL, AL_PY )
      def AL_TEST_RD = align_feature_columns_rdkit.out     

      // Drop NaN en test
      dropnan_rows_test( AL_TEST_MR, OUTDIR_VAL, DN_PY, "final_test", "test" )
      dropnan_rows_test_smile( AL_TEST_RD, OUTDIR_VAL, DN_PY, "final_test_smile", "test" )

      FINAL_TEST_GBM   = dropnan_rows_test.out
      FINAL_TEST_SMILE = dropnan_rows_test_smile.out
  }

emit:
  train = FINAL_TRAIN_GBM
  test  = FINAL_TEST_GBM
  train_smiles = FINAL_TRAIN_SMILE
  test_smiles = FINAL_TEST_SMILE

}