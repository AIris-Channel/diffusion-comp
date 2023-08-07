#!/bin/bash
python generate.py --prompt_path eval_prompts/boy1.json --output_path reg_outputs/boy1
python generate.py --prompt_path eval_prompts/boy2.json --output_path reg_outputs/boy2
python generate.py --prompt_path eval_prompts/girl1.json --output_path reg_outputs/girl1
python generate.py --prompt_path eval_prompts/girl2.json --output_path reg_outputs/girl2
