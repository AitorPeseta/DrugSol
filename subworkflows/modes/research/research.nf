#!/usr/bin/env nextflow
/*
========================================================================================
    Research Subworkflow: Model Training Pipeline
========================================================================================
    Complete machine learning workflow for aqueous solubility prediction:
    1. Data ingestion
    2. Data curation and cleaning
    3. Feature engineering and train/test splitting
    4. Exploratory data analysis
    5. Model training (XGBoost, LightGBM, Chemprop, Linear Regression)
    6. Production model generation
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES AND SUBWORKFLOWS
========================================================================================
*/

// Data processing modules
include { common_ingest as COMMON_INGEST } from '../../tools/common_ingest/common_ingest.nf'
include { curate        as CURATE        } from '../../tools/curate/curate.nf'
include { prepare_data  as PREPARE       } from '../../tools/prepare_data/prepare_data.nf'

// Analysis and training modules
include { analysis      as ANALYSIS      } from '../../tools/analysis/analysis.nf'
include { train_methods as TRAIN_STACK   } from '../../tools/train_methods/train_methods.nf'
include { production    as PRODUCTION    } from '../../tools/production/production.nf'

/*
========================================================================================
    SUBWORKFLOW: RESEARCH MODE
========================================================================================
*/

workflow research {
    take:
        cfg_ch  // Configuration channel containing pipeline parameters

    main:
        // Define output directory structure for research artifacts
        def research_outdir = "${params.outdir}/research"
        def ch_outdir_val   = Channel.value(research_outdir)

        /*
        ================================================================================
            STAGE 1: Data Ingestion
        ================================================================================
            Download and load raw solubility data from multiple public sources
            Output: Unified dataset with standardized format
        */
        COMMON_INGEST( ch_outdir_val )

        def ch_unified_raw = COMMON_INGEST.out.unify

        /*
        ================================================================================
            STAGE 2: Data Curation and Quality Control
        ================================================================================
            Apply filtering criteria:
            - Solvent selection
            - Temperature range constraints
            - Outlier detection and removal
        */
        CURATE( ch_unified_raw, ch_outdir_val )
        
        def ch_curated_data = CURATE.out.output

        /*
        ================================================================================
            STAGE 3: Data Preparation and Feature Engineering
        ================================================================================
            - Molecular standardization and normalization
            - Sample weighting based on temperature distribution
            - Scaffold-based train/test splitting (prevents data leakage)
            - Molecular descriptor calculation (Mordred, RDKit)
            - Feature generation for ML models
        */
        
        // Number of cross-validation iterations for model training
        // User can adjust for more robust estimates (higher) or faster runs (lower)
        def n_iters = params.n_iterations ?: 10
        
        // Random seed for reproducibility
        // Allow users to change for different random splits
        def random_seed = params.random_seed ?: 42
        
        // Feature set selection: "all", "mordred_only", "rdkit_only", "minimal"
        // Users can experiment with different feature combinations
        def feature_set = params.feature_set ?: "all"

        PREPARE(
            ch_curated_data, 
            ch_outdir_val,
            n_iters,
            random_seed,
            feature_set
        )

        // Split prepared data into training and testing channels
        def ch_train_csv    = PREPARE.out.train_gbm      // Mordred descriptors for GBM models
        def ch_test_csv     = PREPARE.out.test_gbm
        def ch_train_smiles = PREPARE.out.train_smiles   // SMILES for GNN/Chemprop
        def ch_test_smiles  = PREPARE.out.test_smiles

        /*
        ================================================================================
            STAGE 4: Exploratory Data Analysis (EDA)
        ================================================================================
            Generate visualizations and statistics:
            - Temperature distribution analysis
            - Molecular weight distribution
            - LogS (solubility) distribution
            - Chemical space coverage
            - Train/test set overlap assessment
        */
        ANALYSIS( ch_train_smiles, ch_test_smiles, ch_outdir_val )

        /*
        ================================================================================
            STAGE 5: Model Training and Optimization
        ================================================================================
            Train ensemble of models in parallel:
            - XGBoost: Gradient boosting with tree-based learning
            - LightGBM: Fast gradient boosting framework
            - Chemprop: Graph neural network for molecular property prediction
            - Linear Regression: Baseline interpretable model
            
            Each model undergoes:
            - Hyperparameter optimization (Optuna-based)
            - Cross-validation
            - Out-of-fold prediction generation
            
            Finally determines optimal ensemble strategy (stacking/blending)
        */
        TRAIN_STACK( 
            ch_train_csv, 
            ch_test_csv, 
            ch_train_smiles, 
            ch_test_smiles, 
            ch_outdir_val 
        )

        // Collect training outputs
        def ch_strategy      = TRAIN_STACK.out.BEST_STRATEGY    // Optimal ensemble strategy
        def ch_oof_all       = TRAIN_STACK.out.OOF_ALL_CONCAT   // Out-of-fold predictions
        def best_params_gbm  = TRAIN_STACK.out.BEST_HP_GBM      // Best hyperparameters for GBM models
        def best_params_gnn  = TRAIN_STACK.out.BEST_HP_GNN      // Best hyperparameters for Chemprop

        /*
        ================================================================================
            STAGE 6: Production Model Generation
        ================================================================================
            Create final production-ready model:
            - Retrain on full dataset with optimal hyperparameters
            - Package model with metadata
            - Generate inference-ready artifacts
            - Save performance metrics and validation results
        */
        
        // Extract file paths from channels for production training
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

/*
========================================================================================
    THE END
========================================================================================
*/