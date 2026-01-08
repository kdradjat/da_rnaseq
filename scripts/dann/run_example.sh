#!/bin/bash

# Simple Training with Domain Adaptation
python dann_training.py --source_data_file <source_data_filepath> \
	--source_label_file <source_label_filepath> \
    --target_data_file <target_data_filepath> \
    --target_label_file <target_label_filepath> \
    --batch_size <batch_size> \
	--nb_epochs <nb_epochs> \
    --plot_umap \
    --plot_name <umap_file_name> \
    --model_name <output_model_name> \
    --supervised \
    --dropout <dropout_rate> \
    
# Domain Adaptation with target proportion variation
python dann_training_prop.py --source_data_file <source_data_filepath> \
	--source_label_file <source_label_filepath> \
    --target_data_file <target_data_filepath> \
    --target_label_file <target_label_filepath> \
    --test_data_file <test_data_filepath> \
    --test_label_file <test_label_filepath> \
    --supervised \
    --batch_size <batch_size> \
	--nb_epochs <nb_epochs> \
    --dropout <dropout_rate> \
    --output_name <results_output_filename> \
    
# Domain ADaptation with source proportion variation
python dann_training_pretrain_prop.py --source_data_file <source_data_filepath> \
	--source_label_file <source_target_filepath> \
    --target_data_file <target_data_filepath> \
    --target_label_file <target_label_filepath> \
    --test_data_file <test_data_filepath> \
    --test_label_file <test_label_filepath> \
    --batch_size <batch_size> \
	--nb_epochs <nb_epochs> \
    --supervised \
    --output_name <results_output_filename> \