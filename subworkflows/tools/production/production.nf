nextflow.enable.dsl = 2

include { train_full_gbm }       from '../../../modules/train_full_gbm/train_full_gbm.nf'
include { train_full_chemprop }  from '../../../modules/train_full_chemprop/train_full_chemprop.nf'
include { train_full_tpsa }      from '../../../modules/train_full_tpsa/train_full_tpsa.nf'
include { build_final_ensemble } from '../../../modules/build_final_ensemble/build_final_ensemble.nf'
include { concat_datasets as concat_mordred } from '../../../modules/concat_datasets/concat_datasets.nf'
include { concat_datasets as concat_rdkit }   from '../../../modules/concat_datasets/concat_datasets.nf'
include { publish_resources }      from '../../../modules/publish_resources/publish_resources.nf'


workflow production {
    take:
        path_train_mordred
        path_test_mordred
        path_train_rdkit
        path_test_rdkit
        ch_oof_predictions
        ch_strategy_file
        outdir_val
        path_best_hp_gbm  
        path_best_hp_gnn

    main:
        // 1. LEER ESTRATEGIA
        def ch_strategy_str = ch_strategy_file
            .splitText { it.trim() }
            .map { it }

        // 2. PREPARACIÓN DE DATOS
        def script_concat = Channel.value(file("${baseDir}/bin/concat_datasets.py"))
        def SKIP_PATH = file("${baseDir}/nextflow.config") // Placeholder para skips

        // Concatenar Datasets
        concat_mordred(path_train_mordred, path_test_mordred, script_concat)
        def ch_full_mordred = concat_mordred.out.OUT

        concat_rdkit(path_train_rdkit, path_test_rdkit, script_concat)
        def ch_full_rdkit = concat_rdkit.out.OUT

        publish_resources(ch_full_mordred, ch_full_rdkit)

        // 3. LÓGICA DE BRANCHING (CORREGIDA PARA PARÁMETROS CONSOLIDADOS)
        // ---------------------------------------------------------------
        def needs_gbm  = ['xgb', 'lgbm', 'blend', 'stack']
        def needs_gnn  = ['chemprop', 'gnn', 'blend', 'stack']
        def needs_tpsa = ['tpsa', 'blend', 'stack']

        // A) RAMA GBM
        // Input real actual: [data, strat, hp_consolidated] -> 3 elementos
        def branch_gbm = ch_full_mordred
            .combine(ch_strategy_str)
            .combine(path_best_hp_gbm) 
            // CORRECCIÓN: Quitamos la variable 'id' extra
            .branch { data, strat, hp_dir ->
                run:  strat in needs_gbm
                      // "production" es el ID que asignamos manualmente
                      return tuple("production", data, hp_dir) 
                skip: true
                      return SKIP_PATH
            }
        
        // B) RAMA GNN
        // Input real actual: [data, strat, params_consolidated] -> 3 elementos
        def branch_gnn = ch_full_rdkit
            .combine(ch_strategy_str)
            .combine(path_best_hp_gnn)
            // CORRECCIÓN: Quitamos la variable 'id' extra
            .branch { data, strat, best_params ->
                run:  strat in needs_gnn
                      return tuple("production", data, best_params)
                skip: true
                      return SKIP_PATH
            }

        // C) RAMA TPSA (Sin cambios, no usa params externos)
        def branch_tpsa = ch_full_rdkit.combine(ch_strategy_str)
            .branch { data, strat ->
                run:  strat in needs_tpsa
                      return tuple("production", data)
                skip: true
                      return SKIP_PATH
            }

        // 4. ENTRENAMIENTO
        // ----------------------------
        
        // A. GBM
        def s_gbm = Channel.value(file("${baseDir}/bin/train_full_gbm.py"))
        train_full_gbm(branch_gbm.run, outdir_val, s_gbm)

        // B. Chemprop
        def s_gnn = Channel.value(file("${baseDir}/bin/train_full_chemprop.py"))
        // El proceso espera tuple(id, train, params), outdir, script, tuple(id, checkpoint)
        // Checkpoint vacío para production:
        def ch_ckpt_prod = branch_gnn.run.map { id, tr, p -> tuple(id, []) }

        train_full_chemprop(
            branch_gnn.run, 
            outdir_val, 
            s_gnn
        )

        // C. TPSA
        def s_tpsa = Channel.value(file("${baseDir}/bin/train_full_tpsa.py"))
        train_full_tpsa(branch_tpsa.run, outdir_val, s_tpsa)

        // 5. RECOLECCIÓN (MIX)
        // ----------------------------------
        
        // Mapeo defensivo: Si es lista [id, path] devuelve path. Si es Path (SKIP), devuelve path.
        def result_gbm  = train_full_gbm.out.MODELS_DIR.mix(branch_gbm.skip)
        def path_gbm = result_gbm.map { it instanceof java.util.List ? it[1] : it }
        
        def result_gnn_raw = train_full_chemprop.out.CHEMPROP_DIR.mix(branch_gnn.skip)
        def path_gnn = result_gnn_raw.map { it instanceof java.util.List ? it[1] : it }
        
        def result_tpsa_raw = train_full_tpsa.out.TPSA_MODEL.mix(branch_tpsa.skip)
        def path_tpsa = result_tpsa_raw.map { it instanceof java.util.List ? it[1] : it }

        // 6. CONSTRUCCIÓN FINAL
        // ---------------------
        def s_ens = Channel.value(file("${baseDir}/bin/build_final_ensemble.py"))
        
        build_final_ensemble(
            ch_strategy_str, 
            ch_oof_predictions,
            path_gbm,
            path_gnn,
            path_tpsa,
            outdir_val,
            s_ens,
            path_train_mordred
        )
}