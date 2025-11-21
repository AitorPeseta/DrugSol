nextflow.enable.dsl = 2

// === Includes de módulos ===
include { filter_water }                                    from '../../../modules/filter_water/filter_water.nf'
include { detect_outliers }                                 from '../../../modules/detect_outliers/detect_outliers.nf'
include { filter_outlier }                                  from '../../../modules/filter_outlier/filter_outlier.nf'
include { filter_by_temperature_range }                     from '../../../modules/filter_by_temperature_range/filter_by_temperature_range.nf'


/**
 * Workflow: curate
 * Toma un dataset unificado, añade flags, limpia, hace split y prepara features train/test alineadas.
 */
workflow curate {

  take:
    UNIFIED_CH    // tabla unificada (parquet/csv)
    OUTDIR_VAL    // val con el outdir 

  main:

    // --- 0) Rutas a scripts Python como valores (NO encadenar operadores sobre value) ---
    def FW_PY = Channel.value( file("${baseDir}/bin/filter_water.py") )
    def DET_PY = Channel.value( file("${baseDir}/bin/detect_outliers.py") )
    def FO_PY = Channel.value( file("${baseDir}/bin/filter_outlier.py") )
    def FT_PY = Channel.value( file("${baseDir}/bin/filter_by_temperature_range.py") )

    // --- 1) Filtrado ---
    filter_water( UNIFIED_CH, OUTDIR_VAL, FW_PY )
    def FILTER_W = filter_water.out

    filter_by_temperature_range(FILTER_W, OUTDIR_VAL, FT_PY, '25', '49')
    def FILTER_T = filter_by_temperature_range.out

    // --- 2) Outliers (detección + filtrado) ---
    detect_outliers( FILTER_T, OUTDIR_VAL, DET_PY )
    def OUTLIER = detect_outliers.out

    filter_outlier( OUTLIER, OUTDIR_VAL, FO_PY )
    def FILTER_O = filter_outlier.out

emit:
  output = FILTER_O

}
