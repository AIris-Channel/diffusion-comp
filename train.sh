#!/bin/bash
accelerate launch --mixed_precision fp16 --num_processes 8 train.py -d train_data/boy1 -o model_output/boy1 --vae_path model_output/boy1/autoencoder.pth
accelerate launch --mixed_precision fp16 --num_processes 8 train.py -d train_data/boy2 -o model_output/boy2 --vae_path model_output/boy2/autoencoder.pth
accelerate launch --mixed_precision fp16 --num_processes 8 train.py -d train_data/girl1 -o model_output/girl1 --vae_path model_output/girl1/autoencoder.pth
accelerate launch --mixed_precision fp16 --num_processes 8 train.py -d train_data/girl2 -o model_output/girl2 --vae_path model_output/girl2/autoencoder.pth