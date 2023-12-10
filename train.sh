#!/bin/bash
accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d data
