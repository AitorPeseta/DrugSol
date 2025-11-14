nextflow.enable.dsl = 2

/*
 * Raíz del proyecto: delega en el workflow nombrado `drugsol`
 * definido en workflows/drugsol/main.nf
 */
include { drugsol } from './workflows/drugsol/main.nf'

workflow {
  // Simplemente invoca el workflow incluido
  drugsol()
}
