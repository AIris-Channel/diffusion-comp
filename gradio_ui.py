import gradio as gr
import torch
from PIL import Image
from configs.unidiffuserv1 import get_config
from sample import set_seed, stable_diffusion_beta_schedule
import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.uvit_multi_post_ln_v1 import UViT
from libs.caption_decoder import CaptionDecoder
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from ip_adapter.ip_adapter import ImageProjModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from score_utils.face_model import FaceAnalysis
from score import Evaluator
import numpy as np
import time
import einops
import glob, os
from load_model import get_face_image

def load_model(model_path):
    global image_proj_model, device
    image_proj_model.load_state_dict(torch.load(model_path, map_location=device), False)
    image_proj_model.to(device)

def sample_single(text, config, nnet, autoencoder, device):

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

    _n_samples = text.size(0)


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

    z, clip_img = sample_fn(device=device, text=text)

    def unpreprocess(v):
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v
    
    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    return (unpreprocess(decode(z))[0]*255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

def score_task(prompt, image, _image, ref_image):
    image = Image.fromarray(image)
    _image = Image.fromarray(_image)
    ref_image = Image.fromarray(ref_image)

    ev = Evaluator()

    ref_face_emb = ev.get_face_embedding(ref_image)
    image_face_emb = ev.get_face_embedding(image)
    _image_face_emb = ev.get_face_embedding(_image)
    face_score = (image_face_emb @ ref_face_emb.T).mean().item()
    face_max = (ref_face_emb @ ref_face_emb.T).mean().item()
    face_min = (_image_face_emb @ ref_face_emb.T).mean().item()
    face_score = (face_score - face_min) / (face_max - face_min)

    image_reward_max = ev.image_reward.score(prompt, _image)
    image_reward_min = ev.image_reward.score(prompt, ref_image)
    image_reward_score = ev.image_reward.score(prompt, image)
    image_reward_score = (image_reward_score - image_reward_min) / (image_reward_max - image_reward_min)

    return face_score, image_reward_score

def sample_and_score(prompt, ref_image):
    global device, nnet, autoencoder, clip_text_model, _clip_text_model, caption_decoder, clip_image_processor, image_encoder, image_proj_model, face_model, eval_config
    with torch.no_grad():
        ref_face, _ = get_face_image(face_model, ref_image)
        ref_clip_image = clip_image_processor(images=ref_face, return_tensors="pt").pixel_values
        image_embeds = image_encoder(ref_clip_image.to('cuda')).image_embeds
        ip_tokens = image_proj_model(image_embeds).squeeze(0)
        text = clip_text_model.encode(prompt)
        text = caption_decoder.encode_prefix(text).squeeze(0)
        text = torch.cat([ip_tokens, text], dim=0)
        image = sample_single(text, eval_config, nnet, autoencoder, device)

        _text = _clip_text_model.encode(prompt)
        _text = caption_decoder.encode_prefix(_text).squeeze(0)
        _image = sample_single(_text, eval_config, nnet, autoencoder, device)

        score1, score2 = score_task(prompt, image, _image, ref_image)
        score3 = score1 * 2.5 + score2
        return image, score1, score2, score3

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    eval_config = get_config()
    autoencoder = libs.autoencoder.get_model(**eval_config.autoencoder).to(device)
    clip_text_model = FrozenCLIPEmbedder(version=eval_config.clip_text_model, device=device, max_length=77-eval_config.image_proj_tokens)
    _clip_text_model =  FrozenCLIPEmbedder(version=eval_config.clip_text_model, device=device)
    caption_decoder = CaptionDecoder(device=device, **eval_config.caption_decoder)
    nnet = UViT(**eval_config.nnet).to(device)
    nnet_default_path = './models/uvit_v1.pth'
    nnet.load_state_dict(torch.load(nnet_default_path, map_location=device), False)

    clip_image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained('image_encoder').eval().to(device)
    image_proj_model = ImageProjModel(
        cross_attention_dim=64,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=eval_config.image_proj_tokens,
    ).eval()

    face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_model.prepare(ctx_id=0, det_size=(512, 512))

    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                model_path = gr.Textbox(lines=1, label="Model Path")
                confirm = gr.Button("Change Model")
                confirm.click(load_model, model_path)
                ref_image = gr.Image(label="Reference Image", type="numpy")
                score1 = gr.Number(0, label="Face Score")
                score2 = gr.Number(0, label="Image Reward Score")
                score3 = gr.Number(0, label="Total Score")
            with gr.Column():
                prompt = gr.Textbox(lines=1, label="Prompt")
                generate = gr.Button("Generate")
                image = gr.Image(label="Output Image", type="numpy")
                generate.click(sample_and_score, [prompt, ref_image], [image, score1, score2, score3])

    app.launch(share=True)
