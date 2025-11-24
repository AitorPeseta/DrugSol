nextflow.enable.dsl = 2

// ============================================================================
// MODULE INCLUDES
// ============================================================================
include { filter_water }                from '../../../modules/filter_water/filter_water.nf'
include { filter_by_temperature_range } from '../../../modules/filter_by_temperature_range/filter_by_temperature_range.nf'
include { detect_outliers }             from '../../../modules/detect_outliers/detect_outliers.nf'
include { filter_outlier }              from '../../../modules/filter_outlier/filter_outlier.nf'

/**
 * WORKFLOW: curate
 * ----------------
 * Input: Unified raw dataset.
 * Steps:
 * 1. Filter Water: Keep only aqueous solubility.
 * 2. Filter Temperature: Keep data within [25, 49] °C for physiological relevance + context.
 * 3. Outlier Detection: Identify statistical anomalies.
 * 4. Filter Outliers: Remove the identified anomalies.
 */

workflow curate {

    take:
        ch_unified_data  // Channel: Path to unified.parquet/csv
        outdir_val       // Value: Output directory path

    main:
        // --- Define Scripts (located in bin/) ---
        def script_filter_water   = file("${baseDir}/bin/filter_water.py")
        def script_filter_temp    = file("${baseDir}/bin/filter_by_temperature_range.py")
        def script_detect_outliers= file("${baseDir}/bin/detect_outliers.py")
        def script_filter_outliers= file("${baseDir}/bin/filter_outlier.py")

        // --- 1. Filter Water / Solvents ---
        filter_water(ch_unified_data, outdir_val, script_filter_water)
        
        // --- 2. Filter Temperature Range (25°C - 49°C) ---
        // We capture RTP (25°C) up to high physiological stress (49°C)
        filter_by_temperature_range(
            filter_water.out, 
            outdir_val, 
            script_filter_temp, 
            '25', // Min Temp
            '49'  // Max Temp
        )

        // --- 3. Outlier Management ---
        detect_outliers(filter_by_temperature_range.out, outdir_val, script_detect_outliers)
        
        filter_outlier(detect_outliers.out, outdir_val, script_filter_outliers)

    emit:
        output = filter_outlier.out
}