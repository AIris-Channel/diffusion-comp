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
from transformers import CLIPVisionModelWithProjection
from libs.data import PersonalizedBase
from libs.uvit_multi_post_ln_v1 import UViT
from libs.discriminator import Discriminator
from ip_adapter.ip_adapter import ImageProjModel


def train(config):
    """
    prepare models
    准备各类需要的模型
    """
    # wandb.login()
    # wandb.init(project='diffusion-comp', config=config)
    
    accelerator, device = utils.setup(config)

    train_state = utils.initialize_train_state(config, device, uvit_class=UViT)
    logging.info(f'load nnet from {config.nnet_path}')
    train_state.set_save_target_key(config.save_target_key)
    train_state.nnet.load_state_dict(torch.load(
        config.nnet_path, map_location='cpu'), False)
    # train_state.nnet = train_state.nnet.half()
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    nnet, optimizer = accelerator.prepare(
        train_state.nnet, train_state.optimizer)
    
    
    nnet.to(device)
    lr_scheduler = train_state.lr_scheduler

    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    # autoencoder = autoencoder.half()

    clip_text_model = FrozenCLIPEmbedder(
        version=config.clip_text_model, device=device, max_length=77-config.image_proj_tokens)
    clip_img_model, clip_img_model_preprocess = clip.load(
        config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained('image_encoder').to(device).eval()

    """
    处理数据部分
    """
    # process data
    train_dataset = PersonalizedBase(
        config.data_json, config.data_path, resolution=512, crop_face=True)
    train_dataset.prepare(autoencoder, clip_img_model,
                          clip_text_model, caption_decoder, image_encoder)

    if config.train_text_encoder:
        clip_text_model.to(device)
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
    
    # prepare ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=768,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=config.image_proj_tokens,
    )
    image_proj_model.to(device)
    print(f"Training params: {utils.cnt_params(image_proj_model)}")
    optimizer = utils.get_optimizer(image_proj_model.parameters(), **config.optimizer)
    
    if config.train_text_encoder and config.train_nnet:
        nnet, clip_text_model, image_proj_model, optimizer, train_dataset_loader, lr_scheduler = accelerator.prepare(
            nnet, clip_text_model, image_proj_model, optimizer, train_dataset_loader, lr_scheduler
        )
    elif config.train_nnet:
        nnet, image_proj_model, optimizer, train_dataset_loader, lr_scheduler = accelerator.prepare(
            nnet, image_proj_model, optimizer, train_dataset_loader, lr_scheduler
        )
    elif config.train_text_encoder:
        clip_text_model, image_proj_model, optimizer, train_dataset_loader, lr_scheduler = accelerator.prepare(
            clip_text_model, image_proj_model, optimizer, train_dataset_loader, lr_scheduler
        )
    else:
        image_proj_model, optimizer, train_dataset_loader, lr_scheduler = accelerator.prepare(image_proj_model, optimizer, train_dataset_loader, lr_scheduler)
        
    
    nnet.requires_grad_(False)
    clip_text_model.requires_grad_(False)
    nnet.eval()
    clip_text_model.eval()
    train_state.image_proj_model = image_proj_model
    
    if config.use_discriminator:
        discriminator = Discriminator(config.data)
        disc_loss = 0
    

    def train_step():   
        metrics = dict()
        z, clip_img, text, image_embeds, data_type = next(train_data_generator)
        z = z.to(device)
        clip_img = clip_img.to(device)
        text = text.to(device)
        image_embeds = image_embeds.to(device)
        data_type = data_type.to(device)

        ip_tokens = image_proj_model(image_embeds)
        text = torch.cat([ip_tokens, text], dim=1).unsqueeze(0)
        text = caption_decoder.encode_prefix(text).squeeze(0)
        
        if config.use_discriminator:
            global disc_loss
            if train_state.step % config.disc_steps == 0:
                disc_loss = discriminator.cal_disc(train_dataset.disc_prompt, config, nnet, clip_text_model, autoencoder, caption_decoder, device)

        with torch.cuda.amp.autocast():
            loss, loss_img, loss_clip_img, loss_text = LSimple_T2I(
                img=z, clip_img=clip_img, text=text, data_type=data_type, nnet=nnet, schedule=schedule, device=device)
            loss = loss.mean()
            if config.use_discriminator:
                loss += disc_loss
            accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        optimizer.zero_grad(set_to_none=True)
        metrics['loss'] = accelerator.gather(
            loss.detach().mean()).mean().item()
        metrics['loss_img'] = accelerator.gather(
            loss_img.detach().mean()).mean().item()
        metrics['loss_clip_img'] = accelerator.gather(
            loss_clip_img.detach().mean()).mean().item()
        if config.use_discriminator:
            metrics['disc_loss'] = disc_loss
        # metrics['scale'] = accelerator.scaler.get_scale()
        metrics['lr'] = train_state.optimizer.param_groups[0]['lr']
        return metrics

    @torch.no_grad()
    @torch.autocast(device_type='cuda')
    def eval(total_step):
        """
        write evaluation code here
        """
        from transformers import CLIPImageProcessor
        from score_utils.face_model import FaceAnalysis
        from load_model import process_one_json
        from score import Evaluator, score
        import json

        output_folder = 'outputs'
        test_json_folder_path = 'train_data/json'
        out_json_dir = 'scores_json'

        config.n_samples = 4
        config.n_iter = 1
        
        autoencoder.to(device)
        clip_text_model.to(device)

        face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_model.prepare(ctx_id=0, det_size=(512, 512))

        ev = Evaluator()

        context = {
            'nnet': nnet,
            'autoencoder': autoencoder,
            'clip_text_model': clip_text_model,
            'caption_decoder': caption_decoder,
            'clip_image_processor': CLIPImageProcessor(),
            'image_encoder': image_encoder,
            'image_proj_model': image_proj_model,
            'face_model': face_model,
            'config': config,
            'device': device
        }

        result_scores = []
        for json_file in os.listdir(test_json_folder_path):
            if not json_file.endswith('.json'):
                continue
            with open(os.path.join(test_json_folder_path, json_file), 'r', encoding='utf-8') as f:
                source_json = json.load(f)
            gen_json = process_one_json(source_json, output_folder, context)
            with open(os.path.join('bound_json_outputs', f"{source_json['id']}.json"), 'r', encoding='utf-8') as f:
                bound_json = json.load(f)
            os.makedirs(out_json_dir, exist_ok=True)
            scores = score(ev, source_json, gen_json, bound_json, out_json_dir)
            score_face = scores['normed_face_ac_scores'] / len(gen_json['images'])
            score_image_reward = scores['normed_image_reward_ac_scores'] / len(gen_json['images'])
            result_score = score_face * 2.5 + score_image_reward
            result_scores.append((source_json['id'], result_score))
        
        torch.cuda.empty_cache()

        # then score
        with open(os.path.join(config.workdir, 'score.txt'), 'a') as f:
            f.write(f'{total_step}\n')
            for k, v in result_scores:
                f.write(f'{k}: {v}\n')
            average_score = sum([v for _, v in result_scores]) / len(result_scores)
            f.write(f'average: {average_score}')
        
    
        clip_text_model.to("cpu")
        autoencoder.to("cpu")
        
        return average_score
        
    def loop():
        log_step = 0
        eval_step = 0
        save_step = config.save_interval

        best_score = float('-inf')
        
        while True:
            image_proj_model.train()
            with accelerator.accumulate(image_proj_model):
                metrics = train_step()

            if accelerator.is_main_process:
                image_proj_model.eval()
                total_step = train_state.step * config.batch_size
                if total_step >= eval_step and config.save_best:
                    current_score = eval(total_step)
                    eval_step += config.eval_interval

                    if current_score > best_score:
                        logging.info(f'saving best ckpts to {config.outdir}...')
                        best_score = current_score
                        train_state.save(os.path.join(
                            config.outdir, 'final.ckpt'))

                if total_step >= log_step:
                    logging.info(utils.dct2str(
                        dict(step=total_step, **metrics)))
                    wandb.log(utils.add_prefix(
                        metrics, 'train'), step=total_step)
                    if config.save_best:
                        wandb.log({'average_score': current_score}, step=total_step)
                    log_step += config.log_interval

      

                if total_step >= save_step:
                    logging.info(f'Save and eval checkpoint {total_step}...')
                    train_state.save(os.path.join(
                        config.ckpt_root, f'{total_step:04}.ckpt'))
                    save_step += config.save_interval

            accelerator.wait_for_everyone()

            if total_step >= config.max_step:
                if not config.save_best:
                    logging.info(f"saving final ckpts to {config.outdir}...")
                    train_state.save(os.path.join(config.outdir, 'final.ckpt')) 
                break

    loop()


def get_args():
    parser = argparse.ArgumentParser()
    # key args
    parser.add_argument('-d', '--data_json_file', type=str,
                        default="train_data/data.json", help="Training data")
    parser.add_argument('-r', '--data_root_path', type=str,
                        default="train_data", help="Training data root path")
    parser.add_argument('-o', "--outdir", type=str,
                        default="model_output", help="output of model")

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
    config.data_json = args.data_json_file
    config.data_path = args.data_root_path

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    config.workdir = os.path.join(
        config.log_dir, f"{config_name}-{now}")
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.meta_dir = os.path.join(config.workdir, "meta")
    config.nnet_path = args.nnet_path
    os.makedirs(config.workdir, exist_ok=True)

    train(config)


if __name__ == "__main__":
    main()
