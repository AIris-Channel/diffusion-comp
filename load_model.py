import torch
import utils
from absl import logging
import os
import json
import wandb
import libs.autoencoder
import clip
from libs.clip import FrozenCLIPEmbedder
from libs.caption_decoder import CaptionDecoder
from torch.utils.data import DataLoader
from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I
import yaml
import datetime
from libs.data import PersonalizedBase
from libs.uvit_multi_post_ln_v1 import UViT
from libs.lora import create_network
from libs.discriminator import Discriminator


def process_one_json(json_data, image_output_path, context={}):
    """
    given a json object, process the task the json describes
    write the json
    """
    train_state = context['train_state']
    autoencoder = context['autoencoder']
    clip_img_model = context['clip_img_model']
    clip_text_model = context['clip_text_model']
    caption_decoder = context['caption_decoder']
    accelerator = context['accelerator']
    device = context['device']
    config = context['config']

    """
    处理数据部分
    """
    # process data
    config = context['config']
    data_name = str(json_data['id'])
    class_word = json_data['class_word']
    config.outdir = os.path.join('model_ouput', data_name)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    config.workdir = os.path.join(
        config.log_dir, f"{data_name}-{now}")
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.meta_dir = os.path.join(config.workdir, "meta")
    os.makedirs(config.workdir, exist_ok=True)

    train_dataset = PersonalizedBase(
        json_data['source_group'], resolution=512, class_word=class_word, crop_face=True, use_blip_caption=config.use_blip_caption)
    train_dataset.prepare(autoencoder, clip_img_model,
                          clip_text_model, caption_decoder)
    
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

    """
    准备模型
    """
    nnet = train_state.nnet
    nnet.to(device)
    lr_scheduler = train_state.lr_scheduler
    # prepare lorann
    lorann = create_network(1.0, config.lora_dim, config.lora_alpha, autoencoder, clip_text_model, nnet, neuron_dropout=config.lora_dropout)
    lorann.to(device)
    lorann.apply_to(clip_text_model, nnet, config.train_text_encoder, config.train_nnet)
    trainable_params = lorann.prepare_optimizer_params(
        config.text_encoder_lr,config.nnet_lr,config.optimizer.lr
    )
    print(f"Training params: {utils.cnt_params(lorann)}")
    optimizer = utils.get_optimizer(trainable_params, **config.optimizer)

    if config.train_text_encoder and config.train_nnet:
        nnet, clip_text_model, lorann, optimizer, train_dataset_loader, lr_scheduler = accelerator.prepare(
            nnet, clip_text_model, lorann, optimizer, train_dataset_loader, lr_scheduler
        )
    elif config.train_nnet:
        nnet, lorann, optimizer, train_dataset_loader, lr_scheduler = accelerator.prepare(
            nnet, lorann, optimizer, train_dataset_loader, lr_scheduler
        )
    elif config.train_text_encoder:
        clip_text_model, lorann, optimizer, train_dataset_loader, lr_scheduler = accelerator.prepare(
            clip_text_model, lorann, optimizer, train_dataset_loader, lr_scheduler
        )
    else:
        lorann, optimizer, train_dataset_loader, lr_scheduler = accelerator.prepare(lorann, optimizer, train_dataset_loader, lr_scheduler)

    nnet.requires_grad_(False)
    clip_text_model.requires_grad_(False)
    nnet.eval()
    clip_text_model.eval()
    lorann.prepare_grad_etc(clip_text_model,nnet)
    train_state.lorann = lorann
    
    if config.use_discriminator:
        discriminator = Discriminator(config.data)
        disc_loss = 0

    def train_step():   
        metrics = dict()
        z, clip_img, text, data_type = next(train_data_generator)
        z = z.to(device)
        clip_img = clip_img.to(device)
        text = text.to(device)
        data_type = data_type.to(device)
        
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

        from configs.sample_config import get_config
        from sample import set_seed, sample
        set_seed(42)
        eval_config = get_config()
        
        eval_config.n_samples = 4
        eval_config.n_iter = 1
        
        autoencoder.to(device)
        clip_text_model.to(device)  
        
        torch.cuda.empty_cache()
         
        # first sample
        task_name = data_name
        eval_config.output_path = os.path.join(image_output_path, task_name)

        # 基于给定的prompt进行生成
        prompts = json_data['caption_list']
        for prompt_index, prompt in enumerate(prompts):
            # 根据训练策略
            prompt = prompt.replace(class_word, f'sks {class_word}')

            eval_config.prompt = prompt
            print("sampling with prompt:", prompt)
            with torch.no_grad():
                sample(prompt_index, eval_config, nnet, clip_text_model, autoencoder, device)
        
        
        # then score
        from score import score_one_task
        scores = score_one_task(json_data['source_group'], prompts, './outputs/', data_name)
        with open(os.path.join(config.workdir, 'score.txt'), 'a') as f:
            f.write(f'{total_step}\n')
            for k, v in scores.items():
                f.write(f'{k}: {v}\n')
        
    
        # clip_text_model.to("cpu")
        autoencoder.to("cpu")
        
        return scores
        
    def loop():
        log_step = 0
        eval_step = 0
        save_step = config.save_interval

        best_score = float('-inf')
        
        while True:
            lorann.train()
            with accelerator.accumulate(lorann):
                metrics = train_step()

            if accelerator.is_main_process:
                lorann.eval()
                total_step = train_state.step * config.batch_size
                if total_step >= eval_step and config.save_best:
                    scores = eval(total_step)
                    eval_step += config.eval_interval
                    
                    current_score = sum([
                        (scores['edit_face'] - config.edit_face_min) / (config.edit_face_max - config.edit_face_min) * config.edit_face_ratio +
                        (scores['edit_text'] - config.edit_text_min) / (config.edit_text_max - config.edit_text_min) * config.edit_text_ratio
                    ])
                    
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
                        wandb.log(utils.add_prefix(
                            scores, 'eval'), step=total_step)
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

    """
    生成图片
    """
    from configs.sample_config import get_config
    from sample import set_seed, sample
    set_seed(42)
    eval_config = get_config()
    
    eval_config.n_samples = 4
    eval_config.n_iter = 1
    
    autoencoder.to(device)
    clip_text_model.to(device)
    
    torch.cuda.empty_cache()

    output_folder = os.path.join(image_output_path, data_name)
    eval_config.output_path = output_folder
    os.makedirs(output_folder, exist_ok=True)

    images = []

    for prompt_index, prompt in enumerate(json_data['caption_list']):
        # 根据训练策略
        eval_config.prompt = prompt.replace(class_word, f'sks {class_word}')
        print("sampling with prompt:", prompt)
        with torch.no_grad():
            sample(prompt_index, eval_config, nnet, clip_text_model, autoencoder, device)
        images.append({
            'prompt': prompt,
            'paths': [os.path.join(output_folder, f'{prompt_index}-{idx:03}.jpg') for idx in range(eval_config.n_samples)]
        })

    
    return {
        "id" : json_data["id"],
        "images" : images
    }


def prepare_context():
    """
    load models from disk, pass other config infos
    """
    from configs.unidiffuserv1 import get_config
    config = get_config()
    config.log_dir = 'logs'
    config.nnet_path = 'models/uvit_v1.pth'

    accelerator, device = utils.setup(config)

    train_state = utils.initialize_train_state(config, device, uvit_class=UViT)
    logging.info(f'load nnet from {config.nnet_path}')
    train_state.set_save_target_key(config.save_target_key)
    train_state.nnet.load_state_dict(torch.load(
        config.nnet_path, map_location='cpu'), False)
    # train_state.nnet = train_state.nnet.half()
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    # autoencoder = autoencoder.half()

    clip_text_model = FrozenCLIPEmbedder(
        version=config.clip_text_model, device=device)
    clip_img_model, clip_img_model_preprocess = clip.load(
        config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)

    return {
        'train_state': train_state,
        'autoencoder': autoencoder,
        'clip_img_model': clip_img_model,
        'clip_text_model': clip_text_model,
        'caption_decoder': caption_decoder,
        'accelerator': accelerator,
        'device': device,
        'config': config
    }


if __name__ == "__main__":
    context = prepare_context()
    for json_file in os.listdir('train_data/json'):
        if not json_file.endswith('.json'):
            continue
        with open(f'train_data/json/{json_file}', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        process_one_json(json_data, 'outputs', context)
