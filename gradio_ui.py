import gradio as gr
import torch
from PIL import Image
from configs.sample_config import get_config
from sample import set_seed, stable_diffusion_beta_schedule
import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.uvit_multi_post_ln_v1 import UViT
from libs.caption_decoder import CaptionDecoder
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from score import Evaluator
import numpy as np
import time
import einops
import glob, os
from libs.lora import create_network_from_weights

def load_model(model_path):
    global device, nnet, autoencoder, clip_text_model
    network,weights = create_network_from_weights(1.0,model_path,autoencoder,clip_text_model,nnet,for_inference=True)
    network.apply_to(clip_text_model,nnet)
    info = network.load_state_dict(weights,False)
    print(f"LoRA weights are loaded: {info}")
    network.to(device)

def sample_single(prompt, config, nnet, clip_text_model, autoencoder, caption_decoder, device):
    print(f"Generating for text: {prompt}")
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

    return (unpreprocess(decode(z))[0]*255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

def score_task(image, prompt, dataset_folder):
    eval = Evaluator()
    # get face, and ref image from dataset folder
    refs1 = glob.glob(os.path.join(dataset_folder, "*.jpg"))
    refs2 = glob.glob(os.path.join(dataset_folder, "*.jpeg"))
    refs = refs1 + refs2
    refs_images = [Image.open(ref) for ref in refs]

    refs_clip = [eval.get_img_embedding(i) for i in refs_images]
    refs_clip = torch.cat(refs_clip)
    # print(refs_clip.shape)

    refs_embs = [eval.get_face_embedding(i) for i in refs_images]
    refs_embs = [emb for emb in refs_embs if emb is not None]
    refs_embs = torch.cat(refs_embs)

    sample = Image.fromarray(image)
    # sample vs ref
    score_face = eval.sim_face_emb(sample, refs_embs)
    score_clip = eval.sim_clip_imgembs(sample, refs_clip)
    # sample vs prompt
    score_text = eval.sim_clip_text(sample, prompt)
    return score_face, score_clip, score_text

def sample_and_score(prompt, dataset_folder):
    global device, nnet, autoencoder, clip_text_model, caption_decoder, eval_config
    with torch.no_grad():
        image = sample_single(prompt, eval_config, nnet, clip_text_model, autoencoder, caption_decoder, device)
        score1, score2, score3 = score_task(image, prompt, f'train_data/{dataset_folder}')
        return image, score1, score2, score3

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    eval_config = get_config()
    autoencoder = libs.autoencoder.get_model(**eval_config.autoencoder).to(device)
    clip_text_model = FrozenCLIPEmbedder(version=eval_config.clip_text_model, device=device)
    caption_decoder = CaptionDecoder(device=device, **eval_config.caption_decoder)
    nnet = UViT(**eval_config.nnet).to(device)
    nnet_default_path = './models/uvit_v1.pth'
    nnet.load_state_dict(torch.load(nnet_default_path, map_location=device), False)
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                model_path = gr.Textbox(lines=1, label="Model Path")
                confirm = gr.Button("Change Model")
                confirm.click(load_model, model_path)
                dataset_folder = gr.Dropdown(["boy1", "boy2", "girl1", "girl2"], label="Dataset Folder")
                score1 = gr.Number(0, label="Face Score")
                score2 = gr.Number(0, label="Clip Image Score")
                score3 = gr.Number(0, label="Clip Text Score")
            with gr.Column():
                prompt = gr.Textbox(lines=1, label="Prompt")
                generate = gr.Button("Generate")
                image = gr.Image(label="Output Image", type="numpy")
                generate.click(sample_and_score, [prompt, dataset_folder], [image, score1, score2, score3])

    app.launch(share=True)
