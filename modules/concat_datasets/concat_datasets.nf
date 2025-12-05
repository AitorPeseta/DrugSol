nextflow.enable.dsl = 2

process concat_datasets {
    tag "Merge Train+Test"
    label 'cpu_small'

    input:
        path train_file
        path test_file
        path script_py

    output:
        path "full_dataset.parquet", emit: OUT

    script:
    """
    python3 ${script_py} \\
        --train ${train_file} \\
        --test ${test_file} \\
        --out full_dataset.parquet
    """
}