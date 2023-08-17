#!/bin/bash
seed=42
python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts/boy1.json --output_path outputs/boy1 --seed $seed
python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts/boy2.json --output_path outputs/boy2 --seed $seed
python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts/girl1.json --output_path outputs/girl1 --seed $seed
python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts/girl2.json --output_path outputs/girl2 --seed $seed
