nextflow.enable.dsl = 2

// ============================================================================
// MODULE INCLUDES
// ============================================================================
// We import the sub-workflows (tools) from their respective locations.
// Aliases (uppercase) help distinguish modules from local variables.
include { common_ingest as COMMON_INGEST } from '../../tools/common_ingest/common_ingest.nf'
include { curate        as CURATE        } from '../../tools/curate/curate.nf'
include { analysis      as ANALYSIS      } from '../../tools/analysis/analysis.nf'
include { prepare_data  as PREPARE       } from '../../tools/prepare_data/prepare_data.nf'
include { train_methods as TRAIN_STACK   } from '../../tools/train_methods/train_methods.nf'
include { production as PRODUCTION   } from '../../tools/production/production.nf'

// ============================================================================
// WORKFLOW DEFINITION
// ============================================================================

workflow research {
    take:
        cfg_ch  // Configuration channel (passed from main.nf)

    main:
        // Define the base output directory for research artifacts
        def research_outdir = "${params.outdir}/research"
        def ch_outdir_val   = Channel.value(research_outdir)

        // ---------------------------------------------------------
        // 1. Data Ingestion
        //    - Download/Load raw data from BigSolDB
        // ---------------------------------------------------------
        COMMON_INGEST( ch_outdir_val )

        def ch_unified_raw = COMMON_INGEST.out.unify

        // ---------------------------------------------------------
        // 2. Curation & Cleaning
        //    - Filter solvents, temperature range and outliers
        // ---------------------------------------------------------
        CURATE( ch_unified_raw, ch_outdir_val )
        
        def ch_curated_data = CURATE.out.output

        // ---------------------------------------------------------
        // 3. Data Preparation & Splitting
        //    - Standardization & Normalization
        //    - Generate Weights based on T
        //    - Scaffold Split (Train/Test)
        //    - Feature Engineering
        // ---------------------------------------------------------
        def n_iters = params.n_iterations ?: 10

        PREPARE(
            ch_curated_data, 
            ch_outdir_val,
            n_iters, 
            42
        )

        def ch_train_csv    = PREPARE.out.train_gbm
        def ch_test_csv     = PREPARE.out.test_gbm
        def ch_train_smiles = PREPARE.out.train_smiles
        def ch_test_smiles  = PREPARE.out.test_smiles
        def ch_standarized  = PREPARE.out.standarized

        // ---------------------------------------------------------
        // 4. Exploratory Data Analysis (EDA)
        //    - Generate plots of distributions (T, MW, LogS)
        //    - Check chemical space coverage
        // ---------------------------------------------------------
        ANALYSIS( ch_train_smiles, ch_test_smiles, ch_outdir_val, ch_standarized )

        // ---------------------------------------------------------
        // 5. Model Training (The Stack)
        //    - Train XGBoost, LightGBM, Chemprop, Regression
        //    - Validate and Ensemble
        // ---------------------------------------------------------
        TRAIN_STACK( 
            ch_train_csv, 
            ch_test_csv, 
            ch_train_smiles, 
            ch_test_smiles, 
            ch_outdir_val 
        )

        def ch_strategy = TRAIN_STACK.out.BEST_STRATEGY
        def ch_oof_all  = TRAIN_STACK.out.OOF_ALL_CONCAT
        def best_params_gbm = TRAIN_STACK.out.BEST_HP_GBM      
        def best_params_gnn = TRAIN_STACK.out.BEST_HP_GNN

        // ---------------------------------------------------------
        // 5. Production
        //    - Produces the ultimate model
        // ---------------------------------------------------------
        train_mordred = ch_train_csv.first().map { id, file -> file }
        test_mordred  = ch_test_csv.first().map  { id, file -> file }
        
        train_rdkit   = ch_train_smiles.first().map { id, file -> file }
        test_rdkit    = ch_test_smiles.first().map  { id, file -> file }
        
        PRODUCTION(
            train_mordred,
            test_mordred,
            train_rdkit,
            test_rdkit,
            ch_oof_all,
            ch_strategy,
            params.outdir,
            best_params_gbm,
            best_params_gnn
        )

}