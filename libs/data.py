from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPImageProcessor
import numpy as np
import os, glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import torch
import gc
from libs.autocrop import get_crop_face_func
from score_utils.face_model import FaceAnalysis


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

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def clip_transform(n_px, crop_face=False):
    if crop_face:
        _crop_face = get_crop_face_func(crop_width=n_px, crop_height=n_px)
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
            _crop_face,
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def vae_transform(resolution,crop_face=False):
    if crop_face:
            _crop_face = get_crop_face_func(crop_width=resolution, crop_height=resolution)
            transform = transforms.Compose([transforms.Resize(resolution), _crop_face, transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    else:
        transform = transforms.Compose([transforms.Resize(resolution), transforms.CenterCrop(resolution),
                                            transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    return transform

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
    return image.crop(best_bbox)


class PersonalizedBase(Dataset):
    def __init__(self,
                 data_dir,
                 resolution,
                 mixing_prob=0.25,
                 crop_face = False,
                 t_drop_rate=0.05,
                 i_drop_rate=0.05,
                 ti_drop_rate=0.05
                ):

        self.data_paths = glob.glob(f'{data_dir}/**/*.jpg')

        self.resolution = resolution
        self.mixing_prob = mixing_prob
        
        self.resolution = resolution
        self.crop_face = crop_face

        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate

        
    def prepare(self, autoencoder, clip_img_model, clip_text_model, caption_decoder, image_encoder, device):
        vae_trans = vae_transform(self.resolution,crop_face=self.crop_face)
        clip_trans = clip_transform(224,crop_face=self.crop_face)
        face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_model.prepare(ctx_id=0, det_size=(512, 512))
        clip_image_processor = CLIPImageProcessor()
        self.empty_text = caption_decoder.encode_prefix(clip_text_model.encode('')).to('cpu')
        for image_path in tqdm(self.data_paths):
            if os.path.exists(image_path + '.pth'):
                continue
            pil_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            text = open(image_path + '.txt','r',encoding='utf-8').read()
            
            img = vae_trans(pil_image)
            img4clip = clip_trans(pil_image)
            
            img = img.to(device).unsqueeze(0)
            img4clip = img4clip.to(device).unsqueeze(0)

            z = autoencoder.encode(img)
            clip_img = clip_img_model.encode_image(img4clip).unsqueeze(1)
            # tokens = clip_text_model.tokenizer.tokenize(text)
            # text_input_ids = clip_text_model.tokenizer.convert_tokens_to_ids(tokens)
            text = clip_text_model.encode(text)
            text = caption_decoder.encode_prefix(text)
            
            face_image = get_face_image(face_model, pil_image)
            if face_image is None:
                face_image = pil_image
            clip_image = clip_image_processor(images=face_image, return_tensors="pt").pixel_values
            image_embeds = image_encoder(clip_image.to(device)).image_embeds.detach().cpu()
            
            data_type = 0
            z = z.to("cpu")
            clip_img = clip_img.to("cpu")
            text = text.to("cpu")
            image_embeds = image_embeds.to("cpu")

            torch.save({
                "z": z,
                "text": text,
                "clip_image": clip_img,
                "data_type": data_type,
                "image_embeds": image_embeds
            }, image_path + '.pth')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, i):
        image_path = self.data_paths[i]
        data_dict = torch.load(image_path + '.pth')

        z = data_dict['z'].squeeze(0)
        clip_img = data_dict['clip_image'].squeeze(0)
        text = data_dict['text'].squeeze(0)
        image_embeds = data_dict['image_embeds'].squeeze(0)
        data_type = data_dict['data_type']

        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            image_embeds = torch.zeros_like(image_embeds)
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = self.empty_text.squeeze(0)
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = self.empty_text.squeeze(0)
            image_embeds = torch.zeros_like(image_embeds, dtype=image_embeds.dtype)
        
        return z, clip_img, text, image_embeds, data_type
