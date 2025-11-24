nextflow.enable.dsl = 2

// ============================================================================
// MODULE INCLUDES
// ============================================================================
include { fetch_bigsoldb } from '../../../modules/fetch_bigsoldb/fetch_bigsoldb.nf'
include { unify_datasets } from '../../../modules/unify_datasets/unify_datasets.nf'

/*
 * WORKFLOW: common_ingest
 * -----------------------
 * Purpose: 
 * 1. Check if user provided an input file (--input).
 * 2. If NOT, download the BigSolDB dataset automatically.
 * 3. Unify/Format the dataset for the next steps.
 */

workflow common_ingest {

    take:
        outdir_val // Value channel: Path to output directory

    main:
        def input_path = params.input
        def source_ch  = null

        // --- Logic: User Input vs. Download ---
        
        // Check if input is provided and not just an empty hyphen or null
        if (input_path && input_path != '-') {
            
            log.info "[Ingest] Using user-provided dataset: ${input_path}"
            source_ch = Channel.fromPath(input_path, checkIfExists: true)

        } else {
            
            log.info "[Ingest] No input provided. Downloading BigSolDB..."
            
            // Define script and parameters
            def script_download = Channel.value(file("${baseDir}/bin/download_bigsoldb.py"))
            def zenodo_record   = Channel.value('15094979') // Explicit version for reproducibility
            
            // Execute Module
            fetch_bigsoldb(outdir_val, zenodo_record, script_download)
            source_ch = fetch_bigsoldb.out
        }

        // --- Logic: Unify/Standardize Data ---
        
        def script_unify = Channel.value(file("${baseDir}/bin/unify_data_sets.py"))
        
        // Pass the source (either user file or downloaded file) to unify
        unify_datasets(source_ch, outdir_val, script_unify)

    emit:
        // Main output: The unified dataset ready for curation
        unify = unify_datasets.out.parquet
}