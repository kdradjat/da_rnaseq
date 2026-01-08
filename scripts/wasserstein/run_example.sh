#!/bin/bash

# Single domain adaptation training
python da_training.py --source_data_file <source_data_filepath> \
	--source_label_file <source_label_filepath> \
    --target_data_file <target_data_filepath> \
    --target_label_file <target_label_filepath> \
    --batch_size <batch_size> \
	--nb_epochs <nb_epochs> \
    --plot_umap \
    --plot_name <umap_file_output_name> \
    --model_name <output_model_name>
    
# Domain Adaptation with target proportion variation
python da_training_prop.py --source_data_file <source_data_filepath> \
	--source_label_file <source_label_filepath> \
    --target_data_file <target_data_filepath> \
    --target_label_file <target_label_filepath> \
    --test_data_file <test_data_filepath> \
    --test_label_file <test_data_filepath> \
    --batch_size <batch_size> \
	--nb_epochs <nb_epochs> \
    --output_name <results_output_filename> \
    
# Domain Adaptation with source proportion variation
python da_training_pretrain_prop.py --source_data_file <source_data_filepath> \
	--source_label_file <source_label_filepath> \
    --target_data_file <target_data_filepath> \
    --target_label_file <target_label_filepath> \
    --test_data_file <test_data_filepath> \
    --test_label_file <test_data_filepath> \
    --batch_size <batch_size> \
	--nb_epochs <nb_epochs> \
    --output_name <results_output_filename> \