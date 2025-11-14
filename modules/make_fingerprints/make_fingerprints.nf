process make_fingerprints {
  tag "make_fingerprints"
  label 'cpu_small'
  publishDir path: { "${outdir}/prepare_data" }, mode: 'copy', overwrite: true

  input:
    path file      
    val  outdir
    path fingerprint_py
    val  name_out

  output:
    path "${name_out}_fingerprint.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/prepare_data"
  YAML="${baseDir}/envs/prepare_data.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${fingerprint_py}" \
                                    -i "${file}" \
                                    --out-parquet "${name_out}_fingerprint.parquet" \
                                    --smiles-col smiles_neutral \
                                    --n-bits 2048 --radius 2 --cluster-cutoff 0.7 \
                                    --save-csv
  """
}