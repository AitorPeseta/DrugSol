process make_features_mordred {
  tag "make_features_mordred"
  label 'cpu_small'
  publishDir path: { "${outdir}/prepare_data" }, mode: 'copy', overwrite: true

  input:
    path file      
    val  outdir
    path features_py
    val  name_out

  output:
    path "${name_out}_mordred_featured.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/prepare_data"
  YAML="${baseDir}/envs/prepare_data.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority
  fi

  # Ejecución directa (Bypass micromamba run)
  "\$PREFIX/bin/python" "${features_py}" --input "${file}" \\
                                    --out_parquet "${name_out}_mordred_featured.parquet" \\
                                    --smiles_source neutral \\
                                    --keep-smiles \\
                                    --include_3d \\
                                    --ff uff \\
                                    --seed_3d 42 \\
                                    --max_atoms_3d 200 \
                                    --max_iters_3d 200 \\
                                    --nproc 4 \\
                                    --inchikey_col cluster_ecfp4_0p7 \\
                                    --ik14_hash_bins 128 \\
                                    --keep_inchikey_as_group \\
                                    --save_csv 

  cp -v "${name_out}_mordred_featured.parquet" "${baseDir}/resources"                               
  """
}