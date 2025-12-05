nextflow.enable.dsl = 2

process cross_validation {
    tag "CV Aggregation"
    label 'cpu_medium'

    conda "${baseDir}/envs/drugsol-data.yml"
    
    publishDir "${params.outdir}/cross_validation", mode: 'copy', overwrite: true

    input:
        path level0_files, stageAs: "level0_*.parquet"
        path blend_files,  stageAs: "blend_*.parquet"
        path stack_files,  stageAs: "stack_*.parquet"
        path metrics,      stageAs: "metrics_*.json"
        path script_py

    output:
        // Salida estructurada para que final_report la entienda
        tuple val("cross_validation"), path("cv_out/test_level0.parquet"), emit: LEVEL0
        tuple val("cross_validation"), path("cv_out/test_blend.parquet"),  emit: BLEND
        tuple val("cross_validation"), path("cv_out/test_stack.parquet"),  emit: STACK, optional: true
        
        path "best_strategy.txt", emit: BEST_STRATEGY
        path "cv_out/metrics_cv_consensus.json"

    script:
    """
    mkdir -p cv_out
    
    # Stack files puede estar vacío, gestionarlo en bash
    # Nota: stack_files es una lista, en bash se expande como 'stack_1.parquet stack_2.parquet ...'
    # Comprobamos si la lista no está vacía
    
    STACK_ARGS=""
    # Si existen archivos que coincidan con el patrón (Nextflow los ha creado)
    if ls stack_*.parquet >/dev/null 2>&1; then
        STACK_ARGS="--stack-files stack_*.parquet"
    fi

    python ${script_py} \\
        --input metrics_*.json \\
        --level0-files level0_*.parquet \\
        --blend-files blend_*.parquet \\
        \$STACK_ARGS \\
        --outdir cv_out
    """
}