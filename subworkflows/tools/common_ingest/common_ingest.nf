nextflow.enable.dsl = 2

// ============================================================================
// MODULE INCLUDES
// ============================================================================
include { fetch_bigsoldb } from '../../../modules/fetch_bigsoldb/fetch_bigsoldb.nf'
include { fetch_chembl } from '../../../modules/fetch_chembl/fetch_chembl.nf'
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
        def all_sources_ch  = null

        // --- Logic: User Input vs. Download ---
        
        // Check if input is provided and not just an empty hyphen or null
        if (input_path && input_path != '-') {
            
            log.info "[Ingest] Using user-provided dataset: ${input_path}"
            all_sources_ch = Channel.fromPath(input_path, checkIfExists: true)

        } else {
            
            log.info "[Ingest] No input provided. Downloading BigSolDB..."
            
            // Define script and parameters
            def script_bigsoldb = Channel.value(file("${baseDir}/bin/download_bigsoldb.py"))
            def zenodo_record   = Channel.value('15094979') // Explicit version for reproducibility
            fetch_bigsoldb(outdir_val, zenodo_record, script_bigsoldb)
            def bigsoldb_ch = fetch_bigsoldb.out

            def script_chembl = Channel.value(file("${baseDir}/bin/download_chembl.py"))
            def ch_chembl_raw = Channel.fromPath("${baseDir}/resources/chembl_raw.csv")
            fetch_chembl(ch_chembl_raw, script_chembl)
            def chembl_ch = fetch_chembl.out

            def extra_source_reaxys_ch = Channel.fromPath("${baseDir}/resources/reaxys.csv", checkIfExists: true)
            def extra_source_challenge_ch = Channel.fromPath("${baseDir}/resources/Second_Challenge_Predict_Aqueous_Solubility.csv", checkIfExists: true)
            all_sources_ch = extra_source_reaxys_ch.mix(bigsoldb_ch, chembl_ch, extra_source_challenge_ch).collect()
        }

        // --- Logic: Unify/Standardize Data ---
        def script_unify = Channel.value(file("${baseDir}/bin/unify_data_sets.py"))
        
        // Pass the source (either user file or downloaded file) to unify
        unify_datasets(all_sources_ch, outdir_val, script_unify)

    emit:
        // Main output: The unified dataset ready for curation
        unify = unify_datasets.out.parquet
}