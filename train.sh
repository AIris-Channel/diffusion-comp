#!/bin/bash
accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 2 train.py -d generate_data
