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
from transformers import CLIPVisionModelWithProjection
from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I
import argparse
import yaml
import datetime
from utils import get_optimizer, get_lr_scheduler
from libs.data import PersonalizedBase
from libs.uvit_multi_post_ln_v1 import UViT
from libs.ip_adapter import ImageProjModel, IPAttnProcessor, IPAdapter


def train(config):
    """
    prepare models
    准备各类需要的模型
    """
    # wandb.login()
    # wandb.init(project='diffusion-comp', config=config)
    
    accelerator, device = utils.setup(config)

    nnet = UViT(**config.nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(
        config.nnet_path, map_location='cpu'), False)
    # train_state.nnet = train_state.nnet.half()
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    nnet.to(device).eval().requires_grad_(False)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    # autoencoder = autoencoder.half()

    clip_text_model = FrozenCLIPEmbedder(
        version=config.clip_text_model, device=device)
    clip_img_model, clip_img_model_preprocess = clip.load(
        config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained('image_encoder').to(device).eval()

    """
    处理数据部分
    """
    # process data
    train_dataset = PersonalizedBase(
        config.data_dir, resolution=512, crop_face=False)
    train_dataset.prepare(autoencoder, clip_img_model,
                          clip_text_model, caption_decoder, image_encoder)
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
        cross_attention_dim=config.ip_cross_attention_dim,
        clip_embeddings_dim=config.ip_clip_embeddings_dim,
        clip_extra_context_tokens=config.image_proj_tokens,
    )

    adapter_modules = torch.nn.ModuleList()
    for blk in nnet.in_blocks:
        attn_proc = IPAttnProcessor(**config.attn_proc)
        attn_proc.apply_to(blk.attn)
        adapter_modules.append(attn_proc)
    attn_proc = IPAttnProcessor(**config.attn_proc)
    attn_proc.apply_to(nnet.mid_block.attn)
    adapter_modules.append(attn_proc)
    for blk in nnet.out_blocks:
        attn_proc = IPAttnProcessor(**config.attn_proc)
        attn_proc.apply_to(blk.attn)
        adapter_modules.append(attn_proc)

    ip_adapter = IPAdapter(image_proj_model, adapter_modules)
    ip_adapter.to(device)
    print(f"Training params: {utils.cnt_params(ip_adapter)}")

    optimizer = get_optimizer(ip_adapter.parameters(), **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)
    ip_adapter, optimizer = accelerator.prepare(
        ip_adapter, optimizer)

    config.step = 0

    def train_step():
        metrics = dict()
        z, clip_img, text, image_embeds, data_type = next(train_data_generator)
        z = z.to(device)
        clip_img = clip_img.to(device)
        text = text.to(device)
        image_embeds = image_embeds.to(device)
        data_type = data_type.to(device)
        
        # image projection
        ip_tokens = ip_adapter(image_embeds)

        with torch.cuda.amp.autocast():
            loss, loss_img, loss_clip_img, loss_text = LSimple_T2I(
                img=z, clip_img=clip_img, text=text, ip_tokens=ip_tokens, data_type=data_type, nnet=nnet, schedule=schedule, device=device)
            accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        config.step += 1
        optimizer.zero_grad(set_to_none=True)
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
            'ip_adapter': ip_adapter,
            'face_model': face_model,
            'config': config,
            'device': device
        }

        result_scores = []
        for json_file in os.listdir(test_json_folder_path)[:2]:
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
            score_total = score_face * 2.5 + score_image_reward
            result_scores.append((source_json['id'], score_face, score_image_reward, score_total))
        
        torch.cuda.empty_cache()

        # then score
        with open(os.path.join(config.workdir, 'score.txt'), 'a') as f:
            f.write(f'{total_step}\n')
            for id, face, image_reward, total in result_scores:
                f.write(f'{id}: face:{face} image reward:{image_reward} total:{total}\n')
            total_face = sum([v for _, v, _, _ in result_scores])
            total_image_reward = sum([v for _, _, v, _ in result_scores])
            total_score = total_face * 2.5 + total_image_reward
            score_dict = {
                'face': total_face,
                'image_reward': total_image_reward,
                'total': total_score
            }
            f.write(f'total: {score_dict}')

        clip_text_model.to("cpu")
        autoencoder.to("cpu")
        
        return score_dict
        
    def loop():
        log_step = 0
        eval_step = 0
        save_step = config.save_interval
        
        best_score = float('-inf')
        
        while True:
            ip_adapter.train()
            with accelerator.accumulate(ip_adapter):
                metrics = train_step()

            if accelerator.is_main_process:
                ip_adapter.eval()
                total_step = config.step * config.batch_size
                if total_step >= eval_step:
                    scores = eval(total_step)
                    eval_step += config.eval_interval
                    
                    current_score = scores['total']
                    
                    if config.save_best and current_score > best_score:
                        logging.info(f'saving best ckpts to {config.outdir}...')
                        best_score = current_score
                        os.makedirs(os.path.join(config.outdir, 'final.ckpt'), exist_ok=True)
                        torch.save(ip_adapter.state_dict(), os.path.join(
                            config.outdir, 'final.ckpt', 'ip_adapter.pth'))
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
                    os.makedirs(os.path.join(config.outdir, f'{total_step:04}.ckpt'), exist_ok=True)
                    torch.save(ip_adapter.state_dict(), os.path.join(
                        config.outdir, f'{total_step:04}.ckpt', 'ip_adapter.pth'))
                    save_step += config.save_interval

                if total_step >= config.max_step and not config.save_best:
                    logging.info(f"saving final ckpts to {config.outdir}...")
                    torch.save(ip_adapter.state_dict(), os.path.join(config.outdir, 'final.ckpt', 'ip_adapter.pth'))
                    break

            accelerator.wait_for_everyone()

    loop()


def get_args():
    parser = argparse.ArgumentParser()
    # key args
    parser.add_argument('-d', '--data_dir', type=str,
                        default="train_data/data.json", help="Training data directory")
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
    config.data_dir = args.data_dir

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
