#!/bin/bash
accelerate launch --mixed_precision fp16 --num_processes 8 train.py
