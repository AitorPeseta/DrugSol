process measurement_counts_histogram {
  tag "measurement_counts_histogram"
  label 'cpu_small'
  publishDir path: { "${outdir}/analysis" }, mode: 'copy', overwrite: true

  input:
    path file
    val  outdir
    path counts_py

  output:
    path "meas_dual_combined", emit: COUNTS_DIR
    
  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/analysis"
  YAML="${baseDir}/envs/analysis.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  # Ejecución directa (Bypass micromamba run)
  "\$PREFIX/bin/python" "${counts_py}" --input "${file}" \\
                                    --id_col smiles_original \\
                                    --xminor 0.5 --grid-alpha 0.4\\
                                    --outdir meas_dual_combined

  """
}