process consolidate_params {
    tag "Consolidating HPs"
    label 'process_low'

    input:
    // Cambiamos 'param_?.json' por 'input_?' para que valga tanto para archivos como directorios
    path json_files, stageAs: 'input_?' 
    path script_consolidate

    output:
    path "best_params_consolidated.json", emit: BEST_PARAMS

    script:
    """
    python3 ${script_consolidate} ${json_files} --output best_params_consolidated.json
    """
}