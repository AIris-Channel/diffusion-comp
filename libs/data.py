from PIL import Image
import io
import torchvision.transforms as transforms
import numpy as np
import re
import os
import PIL
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
import gc
import cv2
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import time
import einops
from sample import prepare_contexts, stable_diffusion_beta_schedule
from configs.sample_config import get_config


training_templates_smallest = [
    'photo of a sks {}',
]

reg_templates_smallest = [
    'photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

face_cascade = cv2.CascadeClassifier('libs/haarcascade_frontalface_default.xml')

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _crop_human_face(image):
    image = transforms.Resize(512)(image)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.15,
        minNeighbors = 5,
        minSize = (5,5)
    )
    # if len(faces) == 0:
    #     return image
    # x = y = w = h = 0
    # for xx, yy, ww, hh in faces:
    #     x += xx
    #     y += yy
    #     w = max(w, ww)
    #     h = max(h, hh)
    # x //= len(faces)
    # y //= len(faces)
    if len(faces) != 1:
        return image
    x, y, w, h = faces[0]
    x = max(0, x - w)
    y = max(0, y - h)
    w = min(image.width, w * 3)
    h = min(image.height, h * 3)
    return image.crop((x, y, x+w, y+h))



def _transform(n_px):
    return transforms.Compose([
        _crop_human_face,
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])



class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 resolution,
                 repeats=100,
                 flip_p=0.5,
                 set="train",
                 class_word="dog",
                 per_image_tokens=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 reg = False
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if re.search(r'\.(?:jpe?g|png)$',file_path)]

        self.num_images = len(self.image_paths) + len(imagenet_templates_small)
        self._length = self.num_images

        self.placeholder_token = class_word
        self.resolution = resolution
        self.per_image_tokens = per_image_tokens
        self.mixing_prob = mixing_prob
        
        
        self.transform_clip = _transform(224)
        self.transform = transforms.Compose([_crop_human_face, transforms.Resize(resolution), transforms.CenterCrop(resolution),
                                             transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.reg = reg
        
    def prepare(self,autoencoder,clip_img_model,clip_text_model,caption_decoder,nnet):
        import os
        os.system("gpustat")
        self.datas = []
        for i in range(self.num_images):
            pil_image = Image.open(self.image_paths[i % self.num_images]).convert("RGB")

            placeholder_string = self.placeholder_token
            if self.coarse_class_text:
                placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

            if not self.reg:
                text = random.choice(training_templates_smallest).format(placeholder_string)
            else:
                text = random.choice(reg_templates_smallest).format(placeholder_string)

            # default to score-sde preprocessing
            img = self.transform(pil_image)
            img4clip = self.transform_clip(pil_image)
            
            img = img.to("cuda").unsqueeze(0)
            img4clip = img4clip.to("cuda").unsqueeze(0)
            
            
            with torch.no_grad():
                z = autoencoder.encode(img)
                clip_img = clip_img_model.encode_image(img4clip).unsqueeze(1)
                text = clip_text_model.encode(text)
                text = caption_decoder.encode_prefix(text)
            data_type = 0
            z = z.to("cpu")
            clip_img = clip_img.to("cpu")
            text = text.to("cpu")
            self.datas.append((z,clip_img,text,data_type))
        
        # 原模型生成的数据
        config = get_config()
        device = "cuda"
        for template in imagenet_templates_small:
            text = template.format(self.placeholder_token)
            print(f"Generating for text: {text}")
            text = clip_text_model.encode(text)
            text = caption_decoder.encode_prefix(text)
            data_type = 1

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

            z, clip_img = sample_fn(device='cuda', text=text)
            
            z = z.to('cpu')
            clip_img = clip_img.to('cpu')
            text = text.to('cpu')
            self.datas.append((z,clip_img,text,data_type))
            

        print("从显存中卸载autoencoder,clip_img_model,clip_text_model,caption_decoder")
        autoencoder = autoencoder.to("cpu")
        clip_img_model = clip_img_model.to("cpu")
        clip_text_model = clip_text_model.to("cpu")
        caption_decoder.caption_model = caption_decoder.caption_model.to("cpu")
        del caption_decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        os.system("gpustat")
        
        

    def __len__(self):
        return self._length

    def __getitem__(self, i):

        z = self.datas[ i % self.num_images ][0].squeeze(0)
        clip_img = self.datas[ i % self.num_images ][1].squeeze(0)
        text = self.datas[ i % self.num_images ][2].squeeze(0)
        data_type = self.datas[ i % self.num_images ][3]
        
        return z,clip_img,text,data_type