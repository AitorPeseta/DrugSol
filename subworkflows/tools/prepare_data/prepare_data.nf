nextflow.enable.dsl = 2

// ============================================================================
// INCLUDES
// ============================================================================

// 1. Standardization & Engineering
include { standardize_smiles }          from '../../../modules/standardize_smiles/standardize_smiles.nf'
include { make_fingerprints }           from '../../../modules/make_fingerprints/make_fingerprints.nf'
// Renamed/Expanded module to include Weights + QED + Ionization
include { engineer_features }           from '../../../modules/engineer_features/engineer_features.nf' 

// 2. Splitting
include { stratified_split }            from '../../../modules/stratified_split/stratified_split.nf'

// 3. Feature Calculation (Aliased for clarity in DAG)
include { make_features_mordred as calc_mordred_train } from '../../../modules/make_features_mordred/make_features_mordred.nf'
include { make_features_mordred as calc_mordred_test  } from '../../../modules/make_features_mordred/make_features_mordred.nf'
include { make_features_rdkit   as calc_rdkit_train   } from '../../../modules/make_features_rdkit/make_features_rdkit.nf'
include { make_features_rdkit   as calc_rdkit_test    } from '../../../modules/make_features_rdkit/make_features_rdkit.nf'

// 4. Feature Processing (Cleaning & Alignment)
include { filter_features       as filter_feat_train             } from '../../../modules/filter_features/filter_features.nf'
include { filter_features       as filter_feat_test              } from '../../../modules/filter_features/filter_features.nf'
include { align_feature_columns as align_mordred                 } from '../../../modules/align_feature_columns/align_feature_columns.nf'
include { align_feature_columns as align_rdkit                   } from '../../../modules/align_feature_columns/align_feature_columns.nf'
include { dropnan_rows          as dropnan_rows_train            } from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows          as dropnan_rows_test             } from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows          as dropnan_rows_train_smile      } from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows          as dropnan_rows_test_smile       } from '../../../modules/dropnan_rows/dropnan_rows.nf'


/**
 * WORKFLOW: prepare_data
 * ----------------------
 * Purpose: 
 * 1. Standardize SMILES.
 * 2. Calculate physical features (Weights, QED, Ionization).
 * 3. Split Data (Research Mode).
 * 4. Generate & Align Descriptors (Mordred/RDKit).
 */
workflow prepare_data {

    take:
        ch_filtered_data   // Input parquet from Curate
        outdir_val         // Output directory

    main: 
        // --- Define Scripts (Binaries) ---
        def script_std     = file("${baseDir}/bin/standardize_smiles.py")
        def script_eng     = file("${baseDir}/bin/engineer_features.py")
        def script_fp      = file("${baseDir}/bin/make_fingerprints.py")
        def script_split   = file("${baseDir}/bin/stratified_split.py")
        def script_mordred = file("${baseDir}/bin/make_features_mordred.py")
        def script_rdkit   = file("${baseDir}/bin/make_features_rdkit.py")
        def script_filter  = file("${baseDir}/bin/filter_features.py")
        def script_align   = file("${baseDir}/bin/align_feature_columns.py")
        def script_drop    = file("${baseDir}/bin/dropnan_rows.py")

        // ============================================================
        // 1. COMMON PRE-PROCESSING
        // ============================================================

        // A. Standardize SMILES (Canonicalize, Salts, etc.)
        standardize_smiles(ch_filtered_data, outdir_val, script_std)
        
        // B. Feature Engineering (The "Physics" Step)
        // Calculates: Gaussian Weights, QED (Drug-likeness), Ionization
        engineer_features(standardize_smiles.out, outdir_val, script_eng)
        
        // C. Fingerprints (needed for Stratified Split)
        make_fingerprints(engineer_features.out, outdir_val, script_fp, "cluster_ecfp4_0p7")
        def ch_ready_to_split = make_fingerprints.out


        // ============================================================
        // 2. BRANCHING LOGIC
        // ============================================================
        
        // Initialize empty channels for outputs
        def ch_final_train_gbm   = Channel.empty()
        def ch_final_test_gbm    = Channel.empty()
        def ch_final_train_smile = Channel.empty()
        def ch_final_test_smile  = Channel.empty() 

        if (params.mode == 'research') {
            
            // --- RESEARCH MODE: SPLIT TRAIN/TEST ---

            // 1. Stratified Scaffold Split
            stratified_split(ch_ready_to_split, outdir_val, script_split)
            def ch_train = stratified_split.out.train
            def ch_test  = stratified_split.out.test

            // ----------------------------------------------------
            // PATH A: GBM Models (Mordred Descriptors)
            // ----------------------------------------------------
            
            // Calculate Raw Features
            calc_mordred_train(ch_train, outdir_val, script_mordred, "train")
            calc_mordred_test(ch_test,   outdir_val, script_mordred, "test")

            // Filter Low Variance / Correlated Features
            filter_feat_train(calc_mordred_train.out, outdir_val, script_filter, "train")
            filter_feat_test(calc_mordred_test.out,   outdir_val, script_filter, "test")

            // Align Test columns to match Train columns exactly
            def ch_mordred_pairs = filter_feat_train.out.combine(filter_feat_test.out)
            align_mordred(ch_mordred_pairs, outdir_val, script_align)

            // Drop NaNs (Final Cleanup)
            dropnan_rows_train(filter_feat_train.out, outdir_val, script_drop, "final_train_gbm", "train")
            dropnan_rows_test(align_mordred.out,      outdir_val, script_drop, "final_test_gbm",  "test")

            ch_final_train_gbm = dropnan_rows_train.out
            ch_final_test_gbm  = dropnan_rows_test.out

            // ----------------------------------------------------
            // PATH B: GNN Models (SMILES + RDKit Features)
            // ----------------------------------------------------
            
            // Calculate RDKit features (lighter than Mordred, often used as auxiliary for GNNs)
            calc_rdkit_train(ch_train, outdir_val, script_rdkit, "train")
            calc_rdkit_test(ch_test,   outdir_val, script_rdkit, "test")

            // Align RDKit columns
            def ch_rdkit_pairs = calc_rdkit_train.out.combine(calc_rdkit_test.out)
            align_rdkit(ch_rdkit_pairs, outdir_val, script_align)

            // Finalizing GNN Inputs (Usually just SMILES + Target + Weight, but we keep features if needed)
            dropnan_rows_train_smile(calc_rdkit_train.out, outdir_val, script_drop, "final_train_gnn", "train")
            dropnan_rows_test_smile(align_rdkit.out,       outdir_val, script_drop, "final_test_gnn",  "test")

            ch_final_train_smile = dropnan_rows_train_smile.out
            ch_final_test_smile  = dropnan_rows_test_smile.out


        } else {
            
            // --- EXECUTION MODE: INFERENCE ONLY ---
            // Input data is treated entirely as "TEST" data.
            // We must load "TRAIN" reference artifacts (column names, scalers) from resources/

            def ch_input_test = ch_ready_to_split

            // 1. Calculate Features
            calc_mordred_test(ch_input_test, outdir_val, script_mordred, "test")
            calc_rdkit_test(ch_input_test,   outdir_val, script_rdkit,   "test")
            
            // 2. Filter (Test only needs to remove its own infinities, column selection happens in align)
            filter_feat_test(calc_mordred_test.out, outdir_val, script_filter, "test")

            // 3. Load Reference Artifacts (Trained Feature Lists)
            def ref_train_mordred = Channel.fromPath("${baseDir}/resources/train_features_mordred_filtered.parquet", checkIfExists: true)
            def ref_train_rdkit   = Channel.fromPath("${baseDir}/resources/train_rdkit_featured.parquet",   checkIfExists: true)

            // 4. Align Test against Reference Train
            align_mordred(ref_train_mordred.combine(filter_feat_test.out), outdir_val, script_align)
            align_rdkit(ref_train_rdkit.combine(calc_rdkit_test.out),      outdir_val, script_align)

            // 5. Drop NaNs
            dropnan_rows_test(align_mordred.out, outdir_val, script_drop, "final_test_gbm", "test")
            dropnan_rows_test_smile(align_rdkit.out, outdir_val, script_drop, "final_test_gnn", "test")

            ch_final_test_gbm   = dropnan_rows_test.out
            ch_final_test_smile = dropnan_rows_test_smile.out
        }

    emit:
        train_gbm    = ch_final_train_gbm
        test_gbm     = ch_final_test_gbm
        train_smiles = ch_final_train_smile
        test_smiles  = ch_final_test_smile
}