#!/bin/bash

python training_prop.py --train_data_file <train_data_filepath> \
	--train_label_file <train_label_filepath> \
    --test_data_file <test_data_filepath> \
    --test_label_file <train_label_filepath> \
	--nb_epochs <nb_epochs> \
    --batch_size <batch_size> \
    --output_name <output_file_name> \