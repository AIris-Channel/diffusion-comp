"""
训练代码
代码输入:
    - 数据文件夹路径, 其中包含近近脸照文件夹和全身照文件夹, 
    - 指定的输出路径, 用于输出模型
    - 其他的参数需要选手自行设定
代码输出:
    - 微调后的模型以及其他附加的子模块
"""

import torch
import utils
from absl import logging
import os
import wandb
import libs.autoencoder
import clip
from libs.clip import FrozenCLIPEmbedder
from libs.caption_decoder import CaptionDecoder
from torch.utils.data import DataLoader
from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I
import argparse
import yaml
import datetime
from pathlib import Path
from libs.data import PersonalizedBase
from libs.uvit_multi_post_ln_v1 import UViT
from utils import get_optimizer, get_lr_scheduler


def train(config):
    """
    prepare models
    准备各类需要的模型
    """
    # wandb.login()
    # wandb.init(project='diffusion-comp', config=config)
    
    accelerator, device = utils.setup(config)

    nnet = UViT(**config.nnet)
    nnet.requires_grad_(False)
    nnet.to(device)
    logging.info(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(
        config.nnet_path, map_location='cpu'), False)

    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)

    clip_text_model = FrozenCLIPEmbedder(
        version=config.clip_text_model, device=device)
    orig_embeds_params = clip_text_model.transformer.text_model.embeddings.token_embedding.weight.detach().clone()
    clip_text_model.transformer.text_model.embeddings.token_embedding.requires_grad_(True)
    token_id = clip_text_model.tokenizer.convert_tokens_to_ids(['sks'])[0]
    print(f"tokens are added: {token_id}")
    index_no_updates = torch.arange(len(clip_text_model.tokenizer)) != token_id

    optimizer = get_optimizer(clip_text_model.transformer.text_model.embeddings.token_embedding.parameters(), **config.optimizer)

    clip_text_model, optimizer = accelerator.prepare(clip_text_model, optimizer)
    
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    clip_img_model, clip_img_model_preprocess = clip.load(
        config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)

    """
    处理数据部分
    """
    # process data
    train_dataset = PersonalizedBase(
        config.data, resolution=512, class_word="boy" if "boy" in config.data else "girl")
    train_dataset.prepare(autoencoder, clip_img_model)
    train_dataset_loader = DataLoader(train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      pin_memory=True,
                                      drop_last=True
                                      )

    train_data_generator = utils.get_data_generator(
        train_dataset_loader, enable_tqdm=accelerator.is_main_process, desc='train')

    logging.info("saving meta data")
    os.makedirs(config.meta_dir, exist_ok=True)
    with open(os.path.join(config.meta_dir, "config.yaml"), "w") as f:
        f.write(yaml.dump(config))
        f.close()

    _betas = stable_diffusion_beta_schedule()
    schedule = Schedule(_betas)
    logging.info(f'use {schedule}')

    step = 0

    def train_step():
        metrics = dict()
        z, clip_img, text, data_type = next(train_data_generator)
        z = z.to(device)
        clip_img = clip_img.to(device)
        text = clip_text_model.encode(text).to(device)
        text = caption_decoder.encode_prefix(text)
        data_type = data_type.to(device)

        with torch.cuda.amp.autocast():
            loss, loss_img, loss_clip_img, loss_text = LSimple_T2I(
                img=z, clip_img=clip_img, text=text, data_type=data_type, nnet=nnet, schedule=schedule, device=device)
            accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        # train_state.ema_update(config.get('ema_rate', 0.9999))
        nonlocal step
        step += 1
        optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            clip_text_model.transformer.text_model.embeddings.token_embedding.weight[index_no_updates] = orig_embeds_params[index_no_updates]

        metrics['loss'] = accelerator.gather(
            loss.detach().mean()).mean().item()
        metrics['loss_img'] = accelerator.gather(
            loss_img.detach().mean()).mean().item()
        metrics['loss_clip_img'] = accelerator.gather(
            loss_clip_img.detach().mean()).mean().item()
        # metrics['scale'] = accelerator.scaler.get_scale()
        metrics['lr'] = optimizer.param_groups[0]['lr']
        return metrics

    @torch.no_grad()
    @torch.autocast(device_type='cuda')
    def eval(total_step):
        """
        write evaluation code here
        """

        from configs.sample_config import get_config
        from sample import set_seed, sample
        import json
        set_seed(42)
        eval_config = get_config()
        for data_name in ['boy1','boy2','girl1','girl2']:
            if data_name in config.workdir:
                eval_config.output_path = os.path.join('outputs', data_name)
                prompt_path = f'eval_prompts/{data_name}.json'
                break
        eval_config.n_samples = 3
        eval_config.n_iter = 1
        
        autoencoder.to(device)
        clip_text_model.to(device)
        
        torch.cuda.empty_cache()
        
        # 基于给定的prompt进行生成
        prompts = json.load(open(prompt_path, "r"))
        for prompt_index, prompt in enumerate(prompts):
            # 根据训练策略
            if "boy" in prompt:
                prompt = prompt.replace("boy", "sks boy")
            else:
                prompt = prompt.replace("girl", "sks girl")

            eval_config.prompt = prompt
            print("sampling with prompt:", prompt)
            with torch.no_grad():
                sample(prompt_index, eval_config, nnet, clip_text_model, autoencoder, device)

        from score import score_one_task
        scores = score_one_task('./train_data/', './eval_prompts/', './outputs/', data_name)
        with open(os.path.join(config.log_dir, 'score.txt'), 'a') as f:
            f.write(f'{total_step}\n')
            for k, v in scores.items():
                f.write(f'{k}: {v}\n')
        print(f"eval score: {scores}")
        
        # clip_text_model.to("cpu")
        autoencoder.to("cpu")
        
        return scores
        
    def loop():
        log_step = 0
        eval_step = 0
        save_step = config.save_interval
        while True:
            nnet.train()
            with accelerator.accumulate(nnet):
                metrics = train_step()

            if accelerator.is_main_process:
                nnet.eval()
                total_step = step * config.batch_size
                if total_step >= eval_step:
                    scores = eval(total_step)
                    eval_step += config.eval_interval
                    
                if total_step >= log_step:
                    logging.info(utils.dct2str(
                        dict(step=total_step, **metrics)))
                    wandb.log(utils.add_prefix(
                        metrics, 'train'), step=total_step)
                    wandb.log(utils.add_prefix(
                        scores, 'eval'), step=total_step)
                    log_step += config.log_interval

      

                if total_step >= save_step:
                    logging.info(f'Save and eval checkpoint {total_step}...')
                    torch.save(clip_text_model.transformer.state_dict(), os.path.join(
                        config.ckpt_root, f'{total_step:04}.pth'))
                    # train_state.save(os.path.join(
                    #     config.ckpt_root, f'{total_step:04}.ckpt'))
                    save_step += config.save_interval

            accelerator.wait_for_everyone()

            if total_step >= config.max_step:
                logging.info(f"saving final ckpts to {config.outdir}...")
                torch.save(clip_text_model.transformer.state_dict(), os.path.join(
                    config.outdir, 'final.pth'))
                # train_state.save(os.path.join(config.outdir, 'final.ckpt'))
                break

    loop()


def get_args():
    parser = argparse.ArgumentParser()
    # key args
    parser.add_argument('-d', '--data', type=str,
                        default="train_data/boy1", help="datadir")
    parser.add_argument('-o', "--outdir", type=str,
                        default="model_ouput/boy1", help="output of model")

    # args of logging
    parser.add_argument("--logdir", type=str, default="logs",
                        help="the dir to put logs")
    parser.add_argument("--nnet_path", type=str,
                        default="models/uvit_v1.pth", help="nnet path to resume")

    return parser.parse_args()


def main():
    # 赛手需要根据自己的需求修改config file
    from configs.unidiffuserv1 import get_config
    config = get_config()
    config_name = "unidiffuserv1"
    args = get_args()
    config.log_dir = args.logdir
    config.outdir = args.outdir
    config.data = args.data
    data_name = Path(config.data).stem

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    config.workdir = os.path.join(
        config.log_dir, f"{config_name}-{data_name}-{now}")
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.meta_dir = os.path.join(config.workdir, "meta")
    config.nnet_path = args.nnet_path
    os.makedirs(config.workdir, exist_ok=True)

    train(config)


if __name__ == "__main__":
    main()
