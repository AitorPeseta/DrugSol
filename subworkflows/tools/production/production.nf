#!/usr/bin/env nextflow
/*
========================================================================================
    Production Subworkflow: Final Model Training and Ensemble Building
========================================================================================
*/

nextflow.enable.dsl = 2

include { train_full_gbm }       from '../../../modules/train_full_gbm/train_full_gbm.nf'
include { train_full_chemprop }  from '../../../modules/train_full_chemprop/train_full_chemprop.nf'
include { train_full_physics }   from '../../../modules/train_full_physics/train_full_physics.nf'
include { build_final_ensemble } from '../../../modules/build_final_ensemble/build_final_ensemble.nf'
include { concat_datasets as concat_mordred } from '../../../modules/concat_datasets/concat_datasets.nf'
include { concat_datasets as concat_rdkit }   from '../../../modules/concat_datasets/concat_datasets.nf'
include { publish_resources }    from '../../../modules/publish_resources/publish_resources.nf'

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
        
        // STAGE 1: Read Best Strategy
        ch_strategy_str = ch_strategy_file
            .splitText()
            .map { it.trim() }
            .first()

        // STAGE 2: Data Preparation
        script_concat = Channel.value(file("${baseDir}/bin/concat_datasets.py"))

        concat_mordred(path_train_mordred, path_test_mordred, script_concat)
        ch_full_mordred = concat_mordred.out.OUT

        concat_rdkit(path_train_rdkit, path_test_rdkit, script_concat)
        ch_full_rdkit = concat_rdkit.out.OUT

        publish_resources(ch_full_mordred, ch_full_rdkit)

        // STAGE 3: Train All Base Models
        
        // GBM
        script_gbm = Channel.value(file("${baseDir}/bin/train_full_gbm.py"))
        input_gbm = ch_full_mordred
            .combine(path_best_hp_gbm)
            .map { mordred_data, hp_dir -> tuple("production", mordred_data, hp_dir) }
        train_full_gbm(input_gbm, outdir_val, script_gbm)

        // Chemprop
        script_gnn = Channel.value(file("${baseDir}/bin/train_full_chemprop.py"))
        input_gnn = ch_full_rdkit
            .combine(path_best_hp_gnn)
            .map { rdkit_data, best_params -> tuple("production", rdkit_data, best_params) }
        train_full_chemprop(input_gnn, outdir_val, script_gnn)

        // Physics
        script_physics = Channel.value(file("${baseDir}/bin/train_full_physics.py"))
        input_physics = ch_full_rdkit
            .map { rdkit_data -> tuple("production", rdkit_data) }
        train_full_physics(input_physics, outdir_val, script_physics)

        // STAGE 4: Collect Model Outputs
        path_gbm = train_full_gbm.out.MODELS_DIR
            .map { meta_id, dir_path -> dir_path }
        
        path_gnn = train_full_chemprop.out.CHEMPROP_DIR
            .map { meta_id, dir_path -> dir_path }
        
        path_physics = train_full_physics.out.MODEL_JSON
            .map { meta_id, json_path -> json_path }

        // STAGE 5: Build Final Ensemble
        script_ensemble = Channel.value(file("${baseDir}/bin/build_final_ensemble.py"))
        
        build_final_ensemble(
            ch_strategy_str, 
            ch_oof_predictions,
            path_gbm,
            path_gnn,
            path_physics,
            outdir_val,
            script_ensemble,
            path_train_mordred
        )
}
