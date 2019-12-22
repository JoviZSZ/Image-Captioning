#!/usr/bin/env python
# coding: utf-8


from datasets import create_input_files


# Create input files (along with word map)
create_input_files(dataset='coco',
                   karpathy_json_path='./dataset_coco.json',
                   image_folder='/datasets/COCO-2015/',
                   captions_per_image=5,
                   min_word_freq=5,
                   output_folder='./inputData/',
                   max_len=50)