import torch
import os
import json
import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.caption_decoder import CaptionDecoder
from libs.uvit_multi_post_ln_v1 import UViT
from libs.ip_adapter import ImageProjModel, IPAttnProcessor, IPAdapter
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image, ImageOps
import einops
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from torchvision.utils import save_image
import numpy as np
from score_utils.face_model import FaceAnalysis


def unpreprocess(v):
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def sample(prompt_index, text, ip_tokens, config, nnet, clip_text_model, autoencoder, caption_decoder, device):
    """
    using_prompt: if use prompt as file name
    """

    n_iter = config.n_iter
    _betas = stable_diffusion_beta_schedule()
    text = torch.stack([text] * config.n_samples)
    ip_tokens = torch.stack([ip_tokens] * config.n_samples)
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


    def t2i_nnet(x, timesteps, text, ip_tokens):  # text is the low dimension version of the text clip embedding
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = split(x)

        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, ip_tokens=ip_tokens, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out = combine(z_out, clip_img_out)

        if config.sample.scale == 0.:
            return x_out

        if config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            _empty_context = caption_decoder.encode_prefix(_empty_context)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=_empty_context, ip_tokens=ip_tokens, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, ip_tokens=ip_tokens, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + config.sample.scale * (x_out - x_out_uncond)


    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)


    _n_samples = text.size(0)


    def sample_fn(**kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
        _x_init = combine(_z_init, _clip_img_init)

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return t2i_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)

        _z, _clip_img = split(x)
        return _z, _clip_img

    samples = None    
    for i in range(n_iter):
        _z, _clip_img = sample_fn(text=text, ip_tokens=ip_tokens)  # conditioned on the text embedding
        new_samples = unpreprocess(decode(_z))
        if samples is None:
            samples = new_samples
        else:
            samples = torch.vstack((samples, new_samples))

    os.makedirs(config.output_path, exist_ok=True)
    for idx, sample in enumerate(samples):
        save_path = os.path.join(config.output_path, f'{prompt_index}-{idx:03}.jpg')
        save_image(sample, save_path)
        
    print(f'results are saved in {save_path}')


def get_face_image(face_model, image):
    bboxes, kpss = face_model.det_model.detect(np.array(image)[:,:,::-1], max_num=1, metric='default')
    if bboxes.shape[0] == 0:
        return None
    best_bbox = bboxes[0, 0:4]
    best_score = bboxes[0, 4]
    for i in range(1, bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        if det_score > best_score:
            best_bbox = bbox
            best_score = det_score
    return image.crop(best_bbox), best_score


def process_one_json(json_data, image_output_path, context={}):
    """
    given a json object, process the task the json describes
    write the json
    """
    nnet = context['nnet']
    autoencoder = context['autoencoder']
    clip_text_model = context['clip_text_model']
    caption_decoder = context['caption_decoder']
    clip_image_processor = context['clip_image_processor']
    image_encoder = context['image_encoder']
    ip_adapter = context['ip_adapter']
    face_model = context['face_model']
    config = context['config']
    device = context['device']
    
    torch.cuda.empty_cache()

    data_name = str(json_data['id'])
    output_folder = os.path.join(image_output_path, data_name)
    config.output_path = output_folder
    os.makedirs(output_folder, exist_ok=True)

    ref_images = [ImageOps.exif_transpose(Image.open(os.path.join('train_data', i['path']))).convert("RGB") for i in json_data['source_group']]
    ref_faces = [get_face_image(face_model, i) for i in ref_images]
    ref_face = max(ref_faces, key=lambda x: x[1])[0]
    ref_clip_image = clip_image_processor(images=ref_face, return_tensors="pt").pixel_values
    image_embeds = image_encoder(ref_clip_image.to('cuda')).image_embeds
    ip_tokens = ip_adapter(image_embeds).squeeze(0)

    images = []

    for prompt_index, prompt in enumerate(json_data['caption_list']):
        print("sampling with prompt:", prompt)
        with torch.no_grad():
            text = clip_text_model.encode(prompt)
            text = caption_decoder.encode_prefix(text).squeeze(0)
            # text = torch.cat([text, ip_tokens], dim=0)
            sample(prompt_index, text, ip_tokens, config, nnet, clip_text_model, autoencoder, caption_decoder, device)

        paths = [os.path.join(output_folder, f'{prompt_index}-{idx:03}.jpg') for idx in range(config.n_samples)]
        
        images.append({
            'prompt': prompt,
            'paths': paths
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
    config.n_samples = 4
    config.n_iter = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init models
    nnet = UViT(**config.nnet).eval()
    print(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location=device), False)
    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    clip_text_model.eval().to(device)
    autoencoder.to(device)
    nnet.to(device)

    clip_image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained('image_encoder').eval().to(device)

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

    ip_adapter = IPAdapter(image_proj_model, adapter_modules).eval()
    state_dict = torch.load('model_output/final.ckpt/ip_adapter.pth', map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    ip_adapter.load_state_dict(state_dict)
    ip_adapter.to(device)

    face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_model.prepare(ctx_id=0, det_size=(512, 512))

    return {
        'nnet': nnet,
        'autoencoder': autoencoder,
        'clip_text_model': clip_text_model,
        'caption_decoder': caption_decoder,
        'clip_image_processor': clip_image_processor,
        'image_encoder': image_encoder,
        'ip_adapter': ip_adapter,
        'face_model': face_model,
        'config': config,
        'device': device
    }


if __name__ == "__main__":
    context = prepare_context()
    for json_file in os.listdir('train_data/json'):
        if not json_file.endswith('.json'):
            continue
        with open(f'train_data/json/{json_file}', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        gen_json = process_one_json(json_data, 'outputs', context)
        with open(f"outputs/{json_data['id']}.json", 'w', encoding='utf-8') as f:
            json.dump(gen_json, f)
