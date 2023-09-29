import argparse
from score import score
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--dataset', type=str, default='./train_data/', help='dataset folder')
    parser.add_argument('--prompts', type=str, default='./eval_prompts_advance/', help='prompt folder')
    parser.add_argument('--outputs', type=str, default='./outputs/', help='output folder')
    parser.add_argument('--task_name', type=str, default='', help='task name')

    args = parser.parse_args()
    
    eval_score = score(args.dataset, args.prompts, args.outputs)
    with open('score_record.txt','a',encoding='utf-8') as f:
        if args.task_name:
            f.write(f'{args.task_name}\n')
        f.write(f'{eval_score}\n\n')
