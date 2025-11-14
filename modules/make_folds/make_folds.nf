process make_folds {
  tag "make_folds"
  label 'cpu_small'
  publishDir path: { "${outdir}/training" }, mode: 'copy', overwrite: true

  input:
    path train
    val  outdir
    path folds_py

  output:
    path "folds.parquet", emit: out

  script:
  """
  set -euo pipefail

  PREFIX="\$HOME/.conda_nf/train_methods"
  YAML="${baseDir}/envs/train_methods.yml"

  if [[ ! -d "\$PREFIX" ]]; then
    rm -rf  "\$PREFIX"
    ${params.MAMBA} create -y -p "\$PREFIX" -f "\$YAML" --strict-channel-priority --always-copy
  fi

  ${params.MAMBA} run -p "\$PREFIX" python "${folds_py}" --input "${train}" \
                                    --out folds.parquet \
                                    --id-col row_uid \
                                    --group-col cluster_ecfp4_0p7 \
                                    --strat-mode both \
                                    --temp-col temperature_K \
                                    --temp-step 2 \
                                    --temp-unit auto \
                                    --target logS --bins 2 \
                                    --n-splits 5 --seed 42
  """
}

