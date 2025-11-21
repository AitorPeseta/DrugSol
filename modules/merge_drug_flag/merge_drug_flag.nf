
process merge_drug_flag {
  tag "merge_drug_flag"
  label 'cpu_small'
  publishDir path: { "${outdir}/curate" }, mode: 'copy', overwrite: true

  input:
    path curated_in        
    val  chembl_csv        
    val  outdir
    val  merge_py

  output:
    path "merged_drup.parquet", emit: out

  script:
  """
  if [ ! -s "${chembl_csv}" ]; then
    echo "[merge] ERROR: chembl_csv vacío o inaccesible" >&2
    exit 1
  fi

  PREFIX="\$HOME/.conda_nf/curate"
  YAML="${baseDir}/envs/curate.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  # [cite_start]Ejecución directa (Bypass micromamba run) [cite: 9]
  "\$PREFIX/bin/python" "${merge_py}" --in "${curated_in}" --chembl "${chembl_csv}" --out "merged_drup.parquet" --export-csv
  """
}