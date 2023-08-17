import datetime
import wandb
from sample import main as sample_main, get_args as get_sample_args
from score import score

def main():
    # 获取当前的日期和时间
    now = datetime.datetime.now()

    # 将日期和时间格式化为字符串
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    # 初始化wandb并设置运行的名称为'score-'加上当前的日期和时间
    wandb.init(project="my-project", name=f"score-{now_str}")

    dataset = './train_data/'
    prompts = './eval_prompts/'
    outputs = './outputs/'

    tasks = ['boy1', 'boy2', 'girl1', 'girl2']
    seeds = [42, 1, 2, 3, 4]

    for seed in seeds:
        for task in tasks:
            # 为每个任务和seed生成样本
            sample_args = get_sample_args()
            sample_args.seed = seed
            sample_args.restore_path = f'model_output/{task}'
            sample_args.prompt_path = f'eval_prompts/{task}.json'
            sample_args.output_path = f'outputs/{task}'
            
            print(sample_args)
            sample_main(sample_args)

            # 对生成的样本进行评分
            eval_score = score(dataset, prompts, outputs)
            print(eval_score)

            # 将结果记录到wandb
            wandb.log({
                "seed": seed, 
                "task": task,
                "face": eval_score["face"], 
                "img_clip": eval_score["img_clip"], 
                "text_clip": eval_score["text_clip"]
            })

if __name__ == "__main__":
    main()