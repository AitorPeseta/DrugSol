nextflow.enable.dsl = 2

include { fetch_bigsoldb }  from '../../../modules/fetch_bigsoldb/fetch_bigsoldb.nf'
include { fetch_chembl }  from '../../../modules/fetch_chembl/fetch_chembl.nf'
include { unify_datasets }  from '../../../modules/unify_datasets/unify_datasets.nf'

/*
 * Versión anterior: SOLO produce UNIFIED
 * - Si --input: lo usa tal cual (csv/parquet)
 * - Si no: descarga BigSolDB y hace unify (de 1 fuente)
 */
workflow common_ingest {

  take:
    OUTDIR_VAL

  main:
    def input_val  = params.input
    def hasInput   = (input_val && input_val.toString().trim() && input_val.toString().trim() != '-')
    def sources_ch 

    //Chembl
    def dl_chem = Channel.value( file("${baseDir}/bin/download_chembl.py") )
    fetch_chembl(OUTDIR_VAL, dl_chem)
    CHEMBL_PATH_CH = fetch_chembl.out          // canal de path (1 elemento)
    CHEMBL_VAL_CH  = CHEMBL_PATH_CH.first()

    if( hasInput ) {
      sources_ch = Channel.fromPath(input_val, checkIfExists:true).collect()
    }
    else {
      // BigSolDB
      def dl_big = Channel.value( file("${baseDir}/bin/download_bigsoldb.py") )
      def rec_id = '15094979'
      fetch_bigsoldb( OUTDIR_VAL, Channel.value(rec_id), dl_big )
      sources_ch = fetch_bigsoldb.out
    }

    def dl_unify = Channel.value( file("${baseDir}/bin/unify_data_sets.py") ) 
    unify_datasets( sources_ch, OUTDIR_VAL, dl_unify ) 
    result_ch = unify_datasets.out
     

  emit:
    unify = result_ch
    chembl = CHEMBL_VAL_CH
}
