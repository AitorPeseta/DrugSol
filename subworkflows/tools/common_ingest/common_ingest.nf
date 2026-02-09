#!/usr/bin/env nextflow
/*
========================================================================================
    Common Ingest Subworkflow: Data Acquisition and Unification
========================================================================================
    Handles data ingestion from multiple sources:
    - User-provided datasets (--input parameter)
    - Automated download from public repositories:
      * BigSolDB (Zenodo)
      * ChEMBL (curated subset)
      * Reaxys proprietary data
      * Challenge datasets
    
    All sources are unified into a standardized format for downstream processing
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { fetch_bigsoldb  } from '../../../modules/fetch_bigsoldb/fetch_bigsoldb.nf'
include { fetch_chembl    } from '../../../modules/fetch_chembl/fetch_chembl.nf'
include { unify_datasets  } from '../../../modules/unify_datasets/unify_datasets.nf'

/*
========================================================================================
    SUBWORKFLOW: COMMON_INGEST
========================================================================================
    Purpose:
    1. Determine data source (user-provided vs. automated download)
    2. Fetch data from selected sources (respecting skip flags)
    3. Unify multiple datasets into standardized format
    
    Input:
    - outdir_val: Output directory for downloaded/processed data
    
    Output:
    - Unified Parquet file containing all solubility measurements
========================================================================================
*/

workflow common_ingest {

    take:
        outdir_val  // Value channel: Path to output directory

    main:
        def input_path      = params.input
        def all_sources_ch  = null

        /*
        ================================================================================
            STAGE 1: Data Source Determination
        ================================================================================
            Two modes of operation:
            A. User-provided dataset: Use --input parameter
            B. Automated download: Fetch from public repositories
        */
        
        // Check if user provided a valid input file path
        if (input_path && input_path != '-') {
            
            log.info "[Ingest] Using user-provided dataset: ${input_path}"
            
            // Validate file existence and create channel
            all_sources_ch = Channel.fromPath(input_path, checkIfExists: true)

        } else {
            
            /*
            ============================================================================
                STAGE 1B: Automated Data Download
            ============================================================================
                Download solubility data from multiple public sources
                Each source can be individually disabled via skip_* parameters
            */
            
            log.info "[Ingest] No input provided. Downloading public datasets..."
            
            // Initialize list to collect active data sources
            def source_channels = []

            // -----------------------------------------------------------------------
            // Source 1: BigSolDB (Primary source - always included)
            // -----------------------------------------------------------------------
            // BigSolDB is the largest open solubility database
            // Zenodo record ID ensures version reproducibility
            
            def script_bigsoldb = Channel.value(file("${baseDir}/bin/fetch_bigsoldb.py"))
            def zenodo_record   = params.bigsoldb_zenodo_id ?: '15094979'
            
            fetch_bigsoldb(outdir_val, zenodo_record, script_bigsoldb)
            source_channels.add(fetch_bigsoldb.out)
            log.info "[Ingest] BigSolDB: ENABLED (Zenodo ID: ${zenodo_record})"

            // -----------------------------------------------------------------------
            // Source 2: ChEMBL (Secondary source - skippable)
            // -----------------------------------------------------------------------
            // ChEMBL provides curated bioactivity data including solubility
            // Pre-filtered subset stored in resources/
            
            if (!params.skip_chembl) {
                def script_chembl = Channel.value(file("${baseDir}/bin/fetch_chembl.py"))
                def ch_chembl_raw = Channel.fromPath("${baseDir}/resources/chembl_raw.csv")
                
                fetch_chembl(ch_chembl_raw, script_chembl)
                source_channels.add(fetch_chembl.out)
                log.info "[Ingest] ChEMBL: ENABLED"
            } else {
                log.info "[Ingest] ChEMBL: SKIPPED (--skip_chembl=true)"
            }

            // -----------------------------------------------------------------------
            // Source 3: Reaxys Proprietary Data (skippable)
            // -----------------------------------------------------------------------
            // Reaxys: Proprietary data (requires local file)
            
            if (!params.skip_reaxys) {
                def reaxys_path = "${baseDir}/resources/reaxys.csv"
                
                // Check if file exists before adding to pipeline
                if (file(reaxys_path).exists()) {
                    def extra_source_reaxys_ch = Channel.fromPath(reaxys_path, checkIfExists: true)
                    source_channels.add(extra_source_reaxys_ch)
                    log.info "[Ingest] Reaxys: ENABLED (${reaxys_path})"
                } else {
                    log.warn "[Ingest] Reaxys: FILE NOT FOUND - skipping (${reaxys_path})"
                }
            } else {
                log.info "[Ingest] Reaxys: SKIPPED (--skip_reaxys=true)"
            }

            // -----------------------------------------------------------------------
            // Source 4: Challenge Dataset (skippable)
            // -----------------------------------------------------------------------
            // Challenge dataset: Public competition data (2019 Solubility Challenge)
            
            if (!params.skip_challenge) {
                def challenge_path = "${baseDir}/resources/Second_Challenge_Predict_Aqueous_Solubility.csv"
                
                // Check if file exists before adding to pipeline
                if (file(challenge_path).exists()) {
                    def extra_source_challenge_ch = Channel.fromPath(challenge_path, checkIfExists: true)
                    source_channels.add(extra_source_challenge_ch)
                    log.info "[Ingest] Challenge: ENABLED (${challenge_path})"
                } else {
                    log.warn "[Ingest] Challenge: FILE NOT FOUND - skipping (${challenge_path})"
                }
            } else {
                log.info "[Ingest] Challenge: SKIPPED (--skip_challenge=true)"
            }

            // -----------------------------------------------------------------------
            // Combine all active data sources
            // -----------------------------------------------------------------------
            // Mix all enabled channels and collect into a single list for unification
            
            if (source_channels.size() == 1) {
                // Only one source - use directly
                all_sources_ch = source_channels[0].collect()
            } else {
                // Multiple sources - mix and collect
                all_sources_ch = source_channels[0]
                for (int i = 1; i < source_channels.size(); i++) {
                    all_sources_ch = all_sources_ch.mix(source_channels[i])
                }
                all_sources_ch = all_sources_ch.collect()
            }
            
            log.info "[Ingest] Total active sources: ${source_channels.size()}"
        }

        /*
        ================================================================================
            STAGE 2: Data Unification
        ================================================================================
            Standardize datasets into common schema:
            - Harmonize column names
            - Convert units to standard format
            - Handle missing values
            - Remove duplicates
            - Export as Parquet for efficient downstream processing
        */
        
        def script_unify = Channel.value(file("${baseDir}/bin/unify_data_sets.py"))
        
        // Unify all source files into single standardized dataset
        unify_datasets(all_sources_ch, outdir_val, script_unify)

    emit:
        // Output: Unified dataset in Parquet format ready for curation
        unify = unify_datasets.out.parquet
}

/*
========================================================================================
    THE END
========================================================================================
*/
