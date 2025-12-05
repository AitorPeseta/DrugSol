process publish_resources {
    tag "Publishing Execution Resources"

    publishDir "${baseDir}/resources", mode: 'copy', overwrite: true

    input:
    path full_mordred, stageAs: 'mordred_source.parquet'
    path full_rdkit,   stageAs: 'rdkit_source.parquet'

    output:
    path "train_features_mordred_filtered.parquet"
    path "train_rdkit_featured.parquet"

    script:
    """
    # Copiamos/Renombramos al nombre final esperado
    cp ${full_mordred} train_features_mordred_filtered.parquet
    cp ${full_rdkit}   train_rdkit_featured.parquet
    """
}