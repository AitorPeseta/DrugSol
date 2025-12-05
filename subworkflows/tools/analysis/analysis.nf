nextflow.enable.dsl = 2

// ============================================================================
// MODULE INCLUDES
// ============================================================================
include { histograms_columns }           from '../../../modules/histograms_columns/histograms_columns.nf'
include { outliers_scatter_mahalanobis } from '../../../modules/outliers_scatter_mahalanobis/outliers_scatter_mahalanobis.nf'
include { measurement_counts_histogram } from '../../../modules/measurement_counts_histogram/measurement_counts_histogram.nf'

/**
 * WORKFLOW: analysis
 * ------------------
 * Performs Exploratory Data Analysis (EDA) on the processed datasets.
 * Generates plots for distributions, outliers, and data counts.
 */
workflow analysis {

    take:
        ch_train_data       // Final Training set (Parquet)
        ch_test_data        // Final Test set (Parquet)
        outdir_val          // Output directory
        ch_curated_raw      // Curated dataset (before feature engineering/splitting)

    main:
        // --- Define Scripts ---
        def script_hist    = file("${baseDir}/bin/histograms_columns.py")
        def script_scatter = file("${baseDir}/bin/outliers_scatter_mahalanobis.py")
        def script_counts  = file("${baseDir}/bin/measurement_counts_histogram.py")

        // Combine Train and Test into a single tuple for comparison plots
        def ch_paired_data = ch_train_data.join(ch_test_data)

        // 1. Compare Distributions (Train vs Test)
        histograms_columns(ch_paired_data, outdir_val, script_hist)

        // 2. Visualize Data Space (PCA + Mahalanobis)
        outliers_scatter_mahalanobis(ch_paired_data, outdir_val, script_scatter)

        // 3. Analyze Redundancy/Measurements per Molecule
        measurement_counts_histogram(ch_curated_raw, outdir_val, script_counts)
}