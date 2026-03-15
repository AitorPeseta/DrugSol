#!/usr/bin/env nextflow
/*
========================================================================================
    Prepare Data Subworkflow: Feature Engineering and Data Splitting
========================================================================================
    Multi-stage data preparation pipeline for machine learning:
    
    Research Mode:
    1. Balance dataset across iterations
    2. Engineer physical/chemical features
    3. Generate molecular fingerprints
    4. Stratified scaffold split (train/test)
    5. Calculate descriptors (Mordred for GBM, RDKit for GNN)
    6. Filter and align features
    7. Final cleanup (drop NaN rows)
    
    Execution Mode:
    1. Engineer features on input data
    2. Calculate descriptors
    3. Align against reference training features
    4. Final cleanup
    
    Dual-path architecture:
    - Path A (GBM): Mordred descriptors for gradient boosting models
    - Path B (GNN): RDKit features + SMILES for graph neural networks
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

// -------------------------------------------------------------------------------------
// 1. Standardization & Feature Engineering
// -------------------------------------------------------------------------------------
include { engineer_features }           from '../../../modules/engineer_features/engineer_features.nf' 
include { make_fingerprints }           from '../../../modules/make_fingerprints/make_fingerprints.nf'

// -------------------------------------------------------------------------------------
// 2. Data Splitting
// -------------------------------------------------------------------------------------
include { balance_dataset }             from '../../../modules/balance_dataset/balance_dataset.nf'
include { stratified_split }            from '../../../modules/stratified_split/stratified_split.nf'

// -------------------------------------------------------------------------------------
// 3. Descriptor Calculation (Aliased for train/test distinction in DAG visualization)
// -------------------------------------------------------------------------------------
include { make_features_mordred as calc_mordred_train  } from '../../../modules/make_features_mordred/make_features_mordred.nf'
include { make_features_mordred as calc_mordred_test   } from '../../../modules/make_features_mordred/make_features_mordred.nf'
include { make_features_rdkit   as calc_rdkit_train    } from '../../../modules/make_features_rdkit/make_features_rdkit.nf'
include { make_features_rdkit   as calc_rdkit_test     } from '../../../modules/make_features_rdkit/make_features_rdkit.nf'

// -------------------------------------------------------------------------------------
// 4. Feature Processing (Filtering, Alignment, Cleanup)
// -------------------------------------------------------------------------------------
include { filter_features       as filter_feat_train        } from '../../../modules/filter_features/filter_features.nf'
include { filter_features       as filter_feat_test         } from '../../../modules/filter_features/filter_features.nf'
include { align_feature_columns as align_mordred            } from '../../../modules/align_feature_columns/align_feature_columns.nf'
include { align_feature_columns as align_rdkit              } from '../../../modules/align_feature_columns/align_feature_columns.nf'
include { dropnan_rows          as dropnan_rows_train       } from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows          as dropnan_rows_test        } from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows          as dropnan_rows_train_smile } from '../../../modules/dropnan_rows/dropnan_rows.nf'
include { dropnan_rows          as dropnan_rows_test_smile  } from '../../../modules/dropnan_rows/dropnan_rows.nf'

/*
========================================================================================
    SUBWORKFLOW: PREPARE_DATA
========================================================================================
    Purpose:
    - Transform curated data into ML-ready features
    - Handle both research (train+test) and execution (inference) modes
    - Generate parallel feature sets for GBM and GNN model architectures
    
    Input:
    - ch_filtered_data: Curated dataset from curation step
    - outdir_val:       Output directory path
    - n_iterations:     Number of cross-validation iterations (research mode)
    - seed_val:         Random seed for reproducibility
    - subset:           Data subset identifier
    
    Output:
    - train_gbm:    Training data with Mordred descriptors (GBM models)
    - test_gbm:     Test data with Mordred descriptors (GBM models)
    - train_smiles: Training data with RDKit features (GNN models)
    - test_smiles:  Test data with RDKit features (GNN models)
========================================================================================
*/

workflow prepare_data {

    take:
        ch_filtered_data   // Curated parquet from curation step
        outdir_val         // Output directory path
        n_iterations       // Number of CV iterations
        seed_val           // Random seed
        subset             // Subset identifier

    main: 
        /*
        ================================================================================
            Script Definitions
        ================================================================================
            Define paths to Python scripts used by each processing module
        */
        def script_eng     = file("${baseDir}/bin/engineer_features.py")
        def script_fp      = file("${baseDir}/bin/make_fingerprints.py")
        def script_balance = file("${baseDir}/bin/balance_dataset.py")
        def script_split   = file("${baseDir}/bin/stratified_split.py")
        def script_mordred = file("${baseDir}/bin/make_features_mordred.py")
        def script_rdkit   = file("${baseDir}/bin/make_features_rdkit.py")
        def script_filter  = file("${baseDir}/bin/filter_features.py")
        def script_align   = file("${baseDir}/bin/align_feature_columns.py")
        def script_drop    = file("${baseDir}/bin/dropnan_rows.py")

        /*
        ================================================================================
            Initialize Output Channels
        ================================================================================
            Empty channels that will be populated based on execution mode
        */
        def ch_final_train_gbm   = Channel.empty()
        def ch_final_test_gbm    = Channel.empty()
        def ch_final_train_smile = Channel.empty()
        def ch_final_test_smile  = Channel.empty() 

        /*
        ================================================================================
            MODE BRANCHING: Research vs Execution
        ================================================================================
        */

        if (params.mode == 'research') {
            
            /*
            ============================================================================
                RESEARCH MODE: Full Training Pipeline
            ============================================================================
                - Multiple iterations for cross-validation
                - Train/test split with stratified scaffold sampling
                - Full feature engineering on both sets
            */
            
            // Create iteration channel for parallel CV runs
            ch_iters = Channel.of(1..n_iterations)
            ch_balance_inputs = ch_iters.combine(ch_filtered_data)

            // -----------------------------------------------------------------------
            // Stage 1: Dataset Balancing
            // -----------------------------------------------------------------------
            // Execute balancing N times in parallel
            // Each iteration uses a different seed (base_seed + iteration_id)
            balance_dataset(ch_balance_inputs, script_balance, seed_val)
            def ch_ready_to_split = balance_dataset.out.balanced_data

            // -----------------------------------------------------------------------
            // Stage 2: Physical Feature Engineering
            // -----------------------------------------------------------------------
            // Calculate molecular properties: MW, ionization state, etc.
            engineer_features(ch_ready_to_split, outdir_val, script_eng)
            def ch_rich_data = engineer_features.out

            // -----------------------------------------------------------------------
            // Stage 3: Molecular Fingerprints
            // -----------------------------------------------------------------------
            // Generate fingerprints for stratified scaffold splitting
            // This ensures chemically similar molecules stay in same split
            make_fingerprints(ch_rich_data, outdir_val, script_fp, 'cluster_ecfp4_0p7')
            def ch_ready_to_stratified = make_fingerprints.out

            // -----------------------------------------------------------------------
            // Stage 4: Stratified Scaffold Split
            // -----------------------------------------------------------------------
            // Split data ensuring scaffold diversity between train/test
            // Prevents data leakage from similar molecular scaffolds
            stratified_split(ch_ready_to_stratified, outdir_val, script_split, seed_val)
            
            // Separate train and test channels with iteration ID
            def train_ch = stratified_split.out.splits.map { id, tr, te -> tuple(id, tr) }
            def test_ch  = stratified_split.out.splits.map { id, tr, te -> tuple(id, te) }

            /*
            ============================================================================
                PATH A: GBM Models (Gradient Boosting with Mordred Descriptors)
            ============================================================================
                Mordred: Comprehensive molecular descriptor calculator
            */
            // Calculate Mordred descriptors for train and test
            calc_mordred_train(train_ch, outdir_val, script_mordred, "train")
            calc_mordred_test(test_ch,   outdir_val, script_mordred, "test")

            // Filter low-variance and highly-correlated features
            filter_feat_train(calc_mordred_train.out, outdir_val, script_filter, "train")
            filter_feat_test(calc_mordred_test.out,   outdir_val, script_filter, "test")

            // Align test columns to match train columns exactly
            // Critical: Ensures model sees identical feature space
            def ch_mordred_pairs = filter_feat_train.out.join(filter_feat_test.out)
            align_mordred(ch_mordred_pairs, outdir_val, script_align)

            // Final cleanup: Remove rows with any remaining NaN values
            dropnan_rows_train(filter_feat_train.out, outdir_val, script_drop, "final_train_gbm", "train", subset)
            dropnan_rows_test(align_mordred.out,      outdir_val, script_drop, "final_test_gbm",  "test", subset)

            ch_final_train_gbm = dropnan_rows_train.out
            ch_final_test_gbm  = dropnan_rows_test.out

            /*
            ============================================================================
                PATH B: GNN Models (Graph Neural Networks with RDKit Features)
            ============================================================================
                RDKit: Lighter descriptor set + SMILES for graph construction
                - SMILES used to build molecular graphs
            */
            
            // Calculate RDKit descriptors
            calc_rdkit_train(train_ch, outdir_val, script_rdkit, "train")
            calc_rdkit_test(test_ch,   outdir_val, script_rdkit, "test")

            // Align RDKit columns between train and test
            def ch_rdkit_pairs = calc_rdkit_train.out.join(calc_rdkit_test.out)
            align_rdkit(ch_rdkit_pairs, outdir_val, script_align)

            // Final cleanup for GNN inputs
            dropnan_rows_train_smile(calc_rdkit_train.out, outdir_val, script_drop, "final_train_gnn", "train", subset)
            dropnan_rows_test_smile(align_rdkit.out,       outdir_val, script_drop, "final_test_gnn",  "test", subset)

            ch_final_train_smile = dropnan_rows_train_smile.out
            ch_final_test_smile  = dropnan_rows_test_smile.out


        } else {
            
            /*
            ============================================================================
                EXECUTION MODE: Inference Pipeline
            ============================================================================
                - Input data treated entirely as "test" data
                - No train/test split (we're predicting on new compounds)
                - Reference training artifacts loaded from resources/
                - Features aligned against pre-computed training features
            */

            // -----------------------------------------------------------------------
            // Stage 1: Physical Feature Engineering
            // -----------------------------------------------------------------------
            def ch_eng_input = ch_filtered_data.map { file -> tuple("test", file) }
            engineer_features(ch_eng_input, outdir_val, script_eng)
            def ch_rich_data = engineer_features.out

            // -----------------------------------------------------------------------
            // Stage 2: Molecular Fingerprints
            // -----------------------------------------------------------------------
            make_fingerprints(ch_rich_data, outdir_val, script_fp, 'cluster_ecfp4_0p7')
            def ch_input_test = make_fingerprints.out
            
            // -----------------------------------------------------------------------
            // Stage 3: Descriptor Calculation
            // -----------------------------------------------------------------------
            calc_mordred_test(ch_input_test, outdir_val, script_mordred, "test")
            calc_rdkit_test(ch_input_test,   outdir_val, script_rdkit,   "test")
            // -----------------------------------------------------------------------
            // Stage 4: Feature Filtering
            // -----------------------------------------------------------------------
            // Test data needs to remove infinities; column selection happens in align
            filter_feat_test(calc_mordred_test.out, outdir_val, script_filter, "test")

            // -----------------------------------------------------------------------
            // Stage 5: Load Reference Training Artifacts
            // -----------------------------------------------------------------------
            // These files define the exact feature columns the trained models expect
            // Critical for ensuring inference uses identical feature space as training
            
            def ref_mordred_path = params.ref_train_mordred ?: "${baseDir}/resources/train_features_mordred_filtered.parquet"
            def ref_rdkit_path   = params.ref_train_rdkit   ?: "${baseDir}/resources/train_rdkit_featured.parquet"
            
            def ref_train_mordred = Channel.fromPath(ref_mordred_path, checkIfExists: true)
            def ref_train_rdkit   = Channel.fromPath(ref_rdkit_path,   checkIfExists: true)

            // -----------------------------------------------------------------------
            // Stage 6: Align Test Against Reference Train
            // -----------------------------------------------------------------------
            // Ensures test data has exactly the same columns as training data
            // Adds missing columns (as zeros) and removes extra columns
            
            align_mordred(
                ref_train_mordred
                    .combine(filter_feat_test.out)
                    .map { train_file, id, test_file -> 
                        tuple(id, train_file, test_file) 
                    }, 
                outdir_val, 
                script_align
            )

            align_rdkit(
                ref_train_rdkit
                    .combine(calc_rdkit_test.out)
                    .map { train_file, id, test_file -> 
                        tuple(id, train_file, test_file) 
                    },      
                outdir_val, 
                script_align
            )

            // -----------------------------------------------------------------------
            // Stage 7: Final Cleanup
            // -----------------------------------------------------------------------
            dropnan_rows_test(align_mordred.out, outdir_val, script_drop, "final_test_gbm", "test", subset)
            dropnan_rows_test_smile(align_rdkit.out, outdir_val, script_drop, "final_test_gnn", "test", subset)

            ch_final_test_gbm   = dropnan_rows_test.out
            ch_final_test_smile = dropnan_rows_test_smile.out
        }

    emit:
        // GBM model inputs (Mordred descriptors)
        train_gbm    = ch_final_train_gbm
        test_gbm     = ch_final_test_gbm
        
        // GNN model inputs (RDKit features + SMILES)
        train_smiles = ch_final_train_smile
        test_smiles  = ch_final_test_smile
}

/*
========================================================================================
    THE END
========================================================================================
*/
