nextflow.enable.dsl = 2

/*
 * Project Root: Entry point delegating execution to the `drugsol` workflow
 * defined in ./workflows/drugsol/main.nf
 */
include { drugsol } from './workflows/drugsol/main.nf'

workflow {
    // Invoke the main workflow
    drugsol()
}