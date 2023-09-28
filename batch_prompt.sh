#!/bin/bash

prompt_list=(', highly detailed, masterpiece' 'b' 'c')

for prompt in "${prompt_list[@]}"; do
    python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_sim.json --output_path outputs/boy1_sim --add_prompt "$prompt"
    python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts_advance/boy2_sim.json --output_path outputs/boy2_sim --add_prompt "$prompt"
    python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_sim.json --output_path outputs/girl1_sim --add_prompt "$prompt"
    python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_sim.json --output_path outputs/girl2_sim --add_prompt "$prompt"
    python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_edit.json --output_path outputs/boy1_edit --add_prompt "$prompt"
    python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts_advance/boy2_edit.json --output_path outputs/boy2_edit --add_prompt "$prompt"
    python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_edit.json --output_path outputs/girl1_edit --add_prompt "$prompt"
    python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_edit.json --output_path outputs/girl2_edit --add_prompt "$prompt"
    python score.py --task_name "$prompt"
done
