nextflow.enable.dsl = 2

// MODULE INCLUDES
include { filter_water }                from '../../../modules/filter_water/filter_water.nf'
include { filter_by_temperature_range } from '../../../modules/filter_by_temperature_range/filter_by_temperature_range.nf'
include { detect_outliers }             from '../../../modules/detect_outliers/detect_outliers.nf'
include { filter_outlier }              from '../../../modules/filter_outlier/filter_outlier.nf'
include { standardize_smiles }          from '../../../modules/standardize_smiles/standardize_smiles.nf'


workflow curate {

    take:
        ch_unified_data
        outdir_val

    main:
        // Definir scripts
        def script_filter_water    = file("${baseDir}/bin/filter_water.py")
        def script_filter_temp     = file("${baseDir}/bin/filter_by_temperature_range.py")
        def script_detect_outliers = file("${baseDir}/bin/detect_outliers.py")
        def script_filter_outliers = file("${baseDir}/bin/filter_outlier.py")
        def script_std             = file("${baseDir}/bin/standardize_smiles.py")

        // --- PASO 1 & 2: Comunes para Research y Execution ---
        
        // 1. Filtrar Agua
        filter_water(ch_unified_data, outdir_val, script_filter_water)
        
        // 2. Filtrar Temperatura
        filter_by_temperature_range(
            filter_water.out, 
            outdir_val, 
            script_filter_temp, 
            '24', '50'
        )

        // Inicializamos el canal de salida final
        def ch_outlier = Channel.empty()

        // --- PASO 3 & 4: Condicionales (Solo Research) ---
        if (params.mode == 'research') {
            
            // En Research: Detectamos y filtramos outliers estadísticos (requiere logS real)
            detect_outliers(filter_by_temperature_range.out, outdir_val, script_detect_outliers)
            filter_outlier(detect_outliers.out, outdir_val, script_filter_outliers)
            
            ch_outlier = filter_outlier.out

        } else {
            ch_outlier = filter_by_temperature_range.out
        }

        standardize_smiles(ch_outlier, outdir_val, script_std)
        def ch_standardized = standardize_smiles.out

    emit:
        output = ch_standardized
}