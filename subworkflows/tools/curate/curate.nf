#!/usr/bin/env nextflow
/*
========================================================================================
    Curate Subworkflow: Data Quality Control and Filtering
========================================================================================
    Multi-stage data curation pipeline:
    1. Solvent filtering (water-only selection)
    2. Temperature range filtering
    3. Statistical outlier detection (research mode only)
    4. Outlier removal (research mode only)
    5. SMILES standardization
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { filter_water                } from '../../../modules/filter_water/filter_water.nf'
include { filter_by_temperature_range } from '../../../modules/filter_by_temperature_range/filter_by_temperature_range.nf'
include { detect_outliers             } from '../../../modules/detect_outliers/detect_outliers.nf'
include { filter_outlier              } from '../../../modules/filter_outlier/filter_outlier.nf'
include { standardize_smiles          } from '../../../modules/standardize_smiles/standardize_smiles.nf'

/*
========================================================================================
    SUBWORKFLOW: CURATE
========================================================================================
    Purpose:
    - Apply quality control filters to raw solubility data
    - Remove unreliable measurements and outliers
    - Standardize molecular representations
    
    Input:
    - ch_unified_data: Unified dataset from ingestion step
    - outdir_val: Output directory for filtered data
    
    Output:
    - Curated dataset ready for feature engineering
========================================================================================
*/

workflow curate {

    take:
        ch_unified_data  // Unified dataset from common_ingest
        outdir_val       // Output directory path

    main:
        // Define Python scripts for each curation step
        def script_filter_water    = file("${baseDir}/bin/filter_water.py")
        def script_filter_temp     = file("${baseDir}/bin/filter_by_temperature_range.py")
        def script_detect_outliers = file("${baseDir}/bin/detect_outliers.py")
        def script_filter_outliers = file("${baseDir}/bin/filter_outlier.py")
        def script_std             = file("${baseDir}/bin/standardize_smiles.py")

        /*
        ================================================================================
            STAGE 1: Solvent Filtering
        ================================================================================
            Filter dataset to include only aqueous solubility measurements
            Removes organic solvents, mixed solvents, and other non-water systems
        */
        filter_water(ch_unified_data, outdir_val, script_filter_water)
        
        /*
        ================================================================================
            STAGE 2: Temperature Range Filtering
        ================================================================================
            Restrict data to physiologically and industrially relevant temperatures
            Default range: 24-50°C (297-323 K)
            
            Rationale:
            - Most biological processes occur in this range
            - Drug formulation typically targets room to body temperature
            - Reduces extrapolation errors in ML models
            
            Users should be able to adjust based on application
        */
        
        // Temperature bounds in Celsius
        // Allow users to define custom temperature ranges
        def temp_min = params.temp_min_celsius ?: '24'
        def temp_max = params.temp_max_celsius ?: '50'
        
        filter_by_temperature_range(
            filter_water.out, 
            outdir_val, 
            script_filter_temp, 
            temp_min,
            temp_max
        )

        // Initialize output channel
        def ch_outlier = Channel.empty()

        /*
        ================================================================================
            STAGE 3 & 4: Outlier Detection and Removal (Research Mode Only)
        ================================================================================
            Statistical outlier detection using:
            - Z-score analysis
            - IQR (Interquartile Range) method
            - Leverage/influence diagnostics
            
            Why research mode only?
            - Requires ground truth logS values for statistical analysis
            - In execution mode, we're predicting unknown compounds
            
            Users may want to adjust outlier sensitivity thresholds
        */
        
        if (params.mode == 'research') {
            
            // Detect statistical outliers in solubility measurements
            detect_outliers(filter_by_temperature_range.out, outdir_val, script_detect_outliers)
            
            // Remove detected outliers from dataset
            filter_outlier(detect_outliers.out, outdir_val, script_filter_outliers)
            
            ch_outlier = filter_outlier.out

        } else {
            // Execution mode: Skip outlier detection, pass data through
            ch_outlier = filter_by_temperature_range.out
        }

        /*
        ================================================================================
            STAGE 5: SMILES Standardization
        ================================================================================
            Normalize molecular representations:
            - Canonical SMILES generation (RDKit)
            - Remove salts and counterions
            - Neutralize charges where appropriate
            - Standardize tautomeric forms
            - Remove stereochemistry if specified
            
            Ensures consistent molecular identity across dataset
        */
        standardize_smiles(ch_outlier, outdir_val, script_std)
        def ch_standardized = standardize_smiles.out

    emit:
        // Output: Curated and standardized dataset
        output = ch_standardized
}

/*
========================================================================================
    THE END
========================================================================================
*/
