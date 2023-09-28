#!/bin/bash

current=0.5
end=1.5
step=0.1

while (( $(awk 'BEGIN {print ('"$current"'<='"$end"')}') ))
do
    python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_sim.json --output_path outputs/boy1_sim --lora_multiplier $current
    python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts_advance/boy2_sim.json --output_path outputs/boy2_sim --lora_multiplier $current
    python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_sim.json --output_path outputs/girl1_sim --lora_multiplier $current
    python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_sim.json --output_path outputs/girl2_sim --lora_multiplier $current
    python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_edit.json --output_path outputs/boy1_edit --lora_multiplier $current
    python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts_advance/boy2_edit.json --output_path outputs/boy2_edit --lora_multiplier $current
    python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_edit.json --output_path outputs/girl1_edit --lora_multiplier $current
    python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_edit.json --output_path outputs/girl2_edit --lora_multiplier $current
    python score.py --task_name $current
    current=$(awk 'BEGIN {print ('"$current"' + '"$step"')}')
done
