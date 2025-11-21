process make_features_rdkit {
  tag "make_features_rdkit"
  label 'cpu_small'
  publishDir path: { "${outdir}/prepare_data" }, mode: 'copy', overwrite: true

  input:
    path file      
    val  outdir
    path features_py
    val  name_out

  output:
    path "${name_out}_rdkit_featured.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/prepare_data"
  YAML="${baseDir}/envs/prepare_data.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  # Ejecución directa (Bypass micromamba run)
  "\$PREFIX/bin/python" "${features_py}" --in "${file}" \\
                                    --out "${name_out}_rdkit_featured.parquet" \\
                                    --smiles-col smiles_neutral

  cp -v "${name_out}_rdkit_featured.parquet" "${baseDir}/resources"                        
  """
}