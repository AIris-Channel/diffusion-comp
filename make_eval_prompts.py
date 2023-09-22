boy_sim = ["a boy"]
girl_sim = ["a girl"]
boy_edit = [
    "a handsom boy",
    "a handsom boy with whole body",
    "a boy standing in the wild",
    "a closer look of boy" ,
    "a boy with glasses",
    "a boy wearing a baseball cap",
    "a boy with a happy expression",
    "a boy in a school uniform",
    "a boy playing soccer",
    "a boy with wavy hair",
    "a boy holding a puppy",
    "a boy at the beach",
    "a boy with his bicycle",
    "a boy in a winter scene",
    "a boy with blue eyes",
    "a boy eating an apple",
    "a boy with a skateboard",
    "a boy in a raincoat",
    "a boy with a backpack",
    "a boy wearing a Halloween costume",
    "a boy with a kite",
    "a boy with freckles",
    "a boy playing a guitar",
    "a boy in a park"
]
girl_edit = [
    "a beautiful girl",
    "a beautiful girl with whole body",
    "a girl standing in the wild",
    "a closer look of girl",
    "a girl with curly hair",
    "a girl wearing a summer dress",
    "a girl with a joyful smile",
    "a girl holding a bunch of flowers",
    "a girl in a ballet pose",
    "a girl with a cat",
    "a girl in a school environment",
    "a girl with a hat",
    "a girl at a music festival",
    "a girl with a teddy bear",
    "a girl with a birthday cake",
    "a girl in a winter sweater",
    "a girl with a book",
    "a girl playing a piano",
    "a girl with a butterfly",
    "a girl in a fairy costume",
    "a girl with a rainbow umbrella",
    "a girl with green eyes",
    "a girl in a kitchen",
    "a girl at the beach"
]

# limit_edit_prompt_nums = 1
limit_edit_prompt_nums = None

if limit_edit_prompt_nums is not None:
    if limit_edit_prompt_nums < len(boy_edit):
        boy_edit = boy_edit[:limit_edit_prompt_nums]
    if limit_edit_prompt_nums < len(girl_edit):
        girl_edit = girl_edit[:limit_edit_prompt_nums]

import os,json

boy1_sim_json_file = os.path.join('./eval_prompts_advance','boy1_sim.json')
boy1_edit_json_file = os.path.join('./eval_prompts_advance','boy1_edit.json')
boy2_edit_json_file = os.path.join('./eval_prompts_advance','boy2_edit.json')
boy2_sim_json_file = os.path.join('./eval_prompts_advance','boy2_sim.json')

with open(boy1_sim_json_file,'w') as f1,open(boy2_sim_json_file,'w') as f2:
    json.dump(boy_sim,f1)
    json.dump(boy_sim,f2)
    
with open(boy1_edit_json_file,'w') as f1,open(boy2_edit_json_file,'w') as f2:
    json.dump(boy_edit,f1)
    json.dump(boy_edit,f2)

girl1_sim_json_file = os.path.join('./eval_prompts_advance','girl1_sim.json')
girl1_edit_json_file = os.path.join('./eval_prompts_advance','girl1_edit.json')
girl2_edit_json_file = os.path.join('./eval_prompts_advance','girl2_edit.json')
girl2_sim_json_file = os.path.join('./eval_prompts_advance','girl2_sim.json')

with open(girl1_sim_json_file,'w') as f1,open(girl2_sim_json_file,'w') as f2:
    json.dump(girl_sim,f1)
    json.dump(girl_sim,f2)
    
with open(girl1_edit_json_file,'w') as f1,open(girl2_edit_json_file,'w') as f2:
    json.dump(girl_edit,f1)
    json.dump(girl_edit,f2)