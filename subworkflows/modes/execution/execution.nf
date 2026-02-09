#!/usr/bin/env nextflow
/*
========================================================================================
    Execution Workflow: Inference Pipeline for New Compounds
========================================================================================
    Applies trained production models to predict aqueous solubility (logS) for 
    new/unknown compounds. This workflow is designed for deployment and routine
    predictions after the research phase has identified optimal models.
    
    Pipeline Stages:
    1. Data Ingestion: Load user-provided compounds or default test set
    2. Curation: Apply same quality filters as training (water, temperature)
    3. Feature Engineering: Calculate Mordred/RDKit descriptors
    4. Prediction: Apply trained ensemble to generate logS predictions
    5. Real Solubility: Convert logS to practical solubility units (mg/mL)
    
    Prerequisites:
    - Research mode must be completed first
    - Production models must exist in results/research/final_product/
    
    Input:
    - User compounds via --input parameter (CSV/Parquet with SMILES column)
    - Or default test dataset if no input provided
    
    Output:
    - Predicted logS values for all input compounds
    - Real solubility in mg/mL at specified pH
    - Confidence intervals (if ensemble used)
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

// -------------------------------------------------------------------------------------
// Data Processing Subworkflows
// -------------------------------------------------------------------------------------
include { common_ingest as COMMON_INGEST } from '../../tools/common_ingest/common_ingest.nf'
include { curate        as CURATE        } from '../../tools/curate/curate.nf'
include { prepare_data  as PREPARE       } from '../../tools/prepare_data/prepare_data.nf'

// -------------------------------------------------------------------------------------
// Prediction Modules
// -------------------------------------------------------------------------------------
include { predict_full_pipeline   } from '../../../modules/predict_full_pipeline/predict_full_pipeline.nf'
include { predict_real_solubility } from '../../../modules/predict_real_solubility/predict_real_solubility.nf'

/*
========================================================================================
    WORKFLOW: EXECUTION
========================================================================================
    Purpose:
    - Apply trained models to predict solubility for new compounds
    - Generate actionable predictions in standard units
    
    Input:
    - cfg_ch: Configuration channel (params passed from main)
    
    Output:
    - Predicted logS and real solubility values
========================================================================================
*/

workflow execution {

    take:
        cfg_ch  // Configuration channel passed from main workflow

    main:
        
        // Define output directory for execution mode results
        def OUTDIR_VAL = Channel.value("${params.outdir}/execution")

        /*
        ================================================================================
            STAGE 1: Data Ingestion
        ================================================================================
            Load compounds for prediction:
            - If --input provided: Use user's compound file
            - Otherwise: Use default test dataset
            
            Supported formats: CSV, Parquet (must contain SMILES column)
        */
        COMMON_INGEST( OUTDIR_VAL )
        def UNIFIED = COMMON_INGEST.out.unify

        /*
        ================================================================================
            STAGE 2: Data Curation
        ================================================================================
            Apply quality control filters (same as training):
            - Filter for aqueous (water) solubility data
            - Apply temperature range filter
            - Standardize SMILES representations
            
            Note: Outlier detection is SKIPPED in execution mode
            (no ground truth logS values available for new compounds)
        */
        CURATE( UNIFIED, OUTDIR_VAL )
        def FINAL_CURATED = CURATE.out.output

        /*
        ================================================================================
            STAGE 3: Feature Engineering
        ================================================================================
            Calculate molecular descriptors for prediction:
            - Mordred descriptors (for GBM models)
            - RDKit features (for Chemprop and Physics baseline)
            
            Features are aligned against training reference to ensure
            identical feature space as production models
        */
        // Note: n_iterations=1 and seed=42 are placeholders (no splitting in execution mode)
        PREPARE( FINAL_CURATED, OUTDIR_VAL, 1, 42, "features_only" )
        
        // Extract feature files (remove iteration ID, not needed in execution)
        def FINAL_TEST_GBM   = PREPARE.out.test_gbm.map    { id, file -> file }
        def FINAL_TEST_SMILE = PREPARE.out.test_smiles.map { id, file -> file }

        /*
        ================================================================================
            STAGE 4: Model Prediction
        ================================================================================
            Apply trained production ensemble to generate predictions:
            - Load models from final_product directory
            - Generate predictions from all base models
            - Apply ensemble strategy (blend/stack) for final prediction
        */
        
        // Locate production model directory
        def final_prod_dir = file("${baseDir}/results/research/final_product/drugsol_model")
        
        // Safety check: Ensure production models exist
        if (!final_prod_dir.exists()) {
            error """
            ================================================================================
            [Execution] ERROR: Production models not found!
            ================================================================================
            Expected location: ${final_prod_dir}
            
            Please run the pipeline in 'research' mode first to train production models:
                nextflow run main.nf --mode research
            
            After research completes, you can run execution mode:
                nextflow run main.nf --mode execution --input your_compounds.csv
            ================================================================================
            """
        }

        def script_predict = Channel.value(file("${baseDir}/bin/predict_full_pipeline.py"))

        predict_full_pipeline(
            FINAL_TEST_GBM,
            FINAL_TEST_SMILE,
            final_prod_dir,
            script_predict
        )

        /*
        ================================================================================
            STAGE 5: Real Solubility Conversion
        ================================================================================
            Convert predicted logS to practical solubility units:
            - Input: logS (log10 of molar solubility)
            - Output: Solubility in mg/mL at specified pH
            
            pH dependency:
            - Ionizable compounds have pH-dependent solubility
            - Default pH 7.4 represents physiological conditions
            - Can be adjusted via params.prediction_ph parameter
        */
        def script_seff = Channel.value(file("${baseDir}/bin/predict_real_solubility.py"))
        
        // pH for solubility calculation (default: physiological pH 7.4)
        def prediction_ph = params.prediction_ph ?: 7.4
        
        predict_real_solubility(
            predict_full_pipeline.out,
            OUTDIR_VAL,
            script_seff,
            prediction_ph
        )
}

/*
========================================================================================
    THE END
========================================================================================
*/
