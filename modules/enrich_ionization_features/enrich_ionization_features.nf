process enrich_ionization_features {
  tag "enrich_ionization_features"
  label 'cpu_small'
  publishDir path: { "${outdir}/prepare_data" }, mode: 'copy', overwrite: true

  input:
    path file      
    val  outdir
    path ionization_py

  output:
    path "ionization.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/prepare_data"
  YAML="${baseDir}/envs/prepare_data.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${ionization_py}" \
                                    --in "${file}" \
                                    --out ionization.parquet \
                                    --smiles-col smiles_neutral \
                                    --smarts ${baseDir}/resources/smarts_pattern_ionized.txt \
                                    --pka-api-url "http://xundrug.cn:5001/modules/upload0/" \
                                    --pka-token "O05DriqqQLlry9kmpCwms2IJLC0MuLQ7"
  """
}

  
