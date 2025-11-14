nextflow.enable.dsl = 2

include { histograms_columns as histogram } from '../../../modules/histograms_columns/histograms_columns.nf'
include { outliers_scatter_mahalanobis as outlier } from '../../../modules/outliers_scatter_mahalanobis/outliers_scatter_mahalanobis.nf'
include { measurement_counts_histogram as measure_counts } from '../../../modules/measurement_counts_histogram/measurement_counts_histogram.nf'


/**
 * Workflow: train_methods
 * Entrena y evalúa modelos de ML (stacking/blending) sobre features ya preparadas.
 */
workflow analysis {

  take:
    FINAL_TRAIN   // tabla train final (parquet/csv)
    FINAL_TEST    // tabla test final (parquet/csv)
    OUTDIR_VAL    // val con el outdir (e.g. Channel.value('results'))
    PARQUET_NOT_FEATURED  // tabla curada sin features (parquet/csv)

  main:

    def HA_PY = Channel.value( file("${baseDir}/bin/histograms_columns.py") )
    histogram(FINAL_TRAIN.combine(FINAL_TEST), OUTDIR_VAL, HA_PY)

    def OSM_PY = Channel.value( file("${baseDir}/bin/outliers_scatter_mahalanobis.py") )
    outlier(FINAL_TRAIN.combine(FINAL_TEST), OUTDIR_VAL, OSM_PY)

    def MCH_PY = Channel.value( file("${baseDir}/bin/measurement_counts_histogram.py") )
    measure_counts(PARQUET_NOT_FEATURED, OUTDIR_VAL, MCH_PY)
}