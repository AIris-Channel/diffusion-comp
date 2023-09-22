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
from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I

from libs.uvit_multi_post_ln_v1 import UViT
from utils import get_optimizer, get_lr_scheduler

from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import time,glob
import einops
import numpy as np
from score import Evaluator
from PIL import Image


class Discriminator:
    def __init__(self,dataset_folder) -> None:
        self.eval = Evaluator()
        refs1 = glob.glob(os.path.join(dataset_folder, "*.jpg"))
        refs2 = glob.glob(os.path.join(dataset_folder, "*.jpeg"))
        refs = refs1 + refs2
        refs_images = [Image.open(ref) for ref in refs]

        refs_clip = [self.eval.get_img_embedding(i) for i in refs_images]
        refs_clip = torch.cat(refs_clip)
        # print(refs_clip.shape)

        refs_embs = [self.eval.get_face_embedding(i) for i in refs_images]
        refs_embs = [emb for emb in refs_embs if emb is not None]
        refs_embs = torch.cat(refs_embs)
        
        self.refs_clip = refs_clip
        self.refs_embs = refs_embs
        
    def cal_disc(self,prompt, config, nnet, clip_text_model, autoencoder, caption_decoder, device):
        image = self.sample_image(prompt, config, nnet, clip_text_model, autoencoder, caption_decoder, device)
        score_face, score_clip, score_text = self.get_score(image, prompt[0])
        print(f"score_face: {score_face}, score_clip: {score_clip}, score_text: {score_text}")
        disc_loss = (1 - score_face) * config.disc_loss_weight
        
        return disc_loss
        
    
    def get_score(self,image,prompt):
        sample = Image.fromarray(image)
        score_face = self.eval.sim_face_emb(sample, self.refs_embs)
        score_clip = self.eval.sim_clip_imgembs(sample, self.refs_clip)
        score_text = self.eval.sim_clip_text(sample, prompt)
        return score_face, score_clip, score_text
    
    def sample_image(self,prompt, config, nnet, clip_text_model, autoencoder, caption_decoder, device):
        prompt = clip_text_model.encode(prompt)
        prompt = caption_decoder.encode_prefix(prompt)
        
        _betas = stable_diffusion_beta_schedule()
        N = len(_betas)

        empty_context = clip_text_model.encode([''])[0]

        def split(x):
            C, H, W = config.z_shape
            z_dim = C * H * W
            z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
            z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
            clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
            return z, clip_img


        def combine(z, clip_img):
            z = einops.rearrange(z, 'B C H W -> B (C H W)')
            clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
            return torch.concat([z, clip_img], dim=-1)


        def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
            """
            1. calculate the conditional model output
            2. calculate unconditional model output
                config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
                config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
            3. return linear combination of conditional output and unconditional output
            """
            z, clip_img = split(x)

            t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

            z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                                data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out = combine(z_out, clip_img_out)

            if config.sample.scale == 0.:
                return x_out

            if config.sample.t2i_cfg_mode == 'empty_token':
                _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
                _empty_context = caption_decoder.encode_prefix(_empty_context)
                z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                        data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
                x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
            elif config.sample.t2i_cfg_mode == 'true_uncond':
                text_N = torch.randn_like(text)  # 3 other possible choices
                z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                        data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
                x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
            else:
                raise NotImplementedError

            return x_out + config.sample.scale * (x_out - x_out_uncond)

        _n_samples = prompt.size(0)


        def sample_fn(device, **kwargs):
            _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
            _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
            _x_init = combine(_z_init, _clip_img_init)

            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

            def model_fn(x, t_continuous):
                t = t_continuous * N
                return t2i_nnet(x, t, **kwargs)

            dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
            with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
                start_time = time.time()
                x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
                end_time = time.time()
                print(f'\ngenerate {_n_samples} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

            _z, _clip_img = split(x)
            return _z, _clip_img

        z, clip_img = sample_fn(device=device, text=prompt)

        def unpreprocess(v):
            v = 0.5 * (v + 1.)
            v.clamp_(0., 1.)
            return v
        
        @torch.cuda.amp.autocast()
        def decode(_batch):
            return autoencoder.decode(_batch)

        image = (unpreprocess(decode(z))[0]*255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        return image
