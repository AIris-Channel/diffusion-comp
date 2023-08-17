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


def _transform(n_px):
    return transforms.Compose([
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

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if not file_path.endswith(".txt")]

        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = class_word
        self.resolution = resolution
        self.per_image_tokens = per_image_tokens
        self.mixing_prob = mixing_prob
        
        
        self.transform_clip = _transform(224)
        self.transform = transforms.Compose([transforms.Resize(resolution), transforms.CenterCrop(resolution),
                                             transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.reg = reg
        
    def prepare(self,autoencoder,clip_img_model,clip_text_model,caption_decoder):
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
        print("从显存中卸载autoencoder,clip_img_model,clip_text_model,caption_decoder")
        autoencoder = autoencoder.to("cpu")
        clip_img_model = clip_img_model.to("cpu")
        clip_text_model = clip_text_model.to("cpu")
        caption_decoder.caption_model = caption_decoder.caption_model.to("cpu")
        del caption_decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        

    def __len__(self):
        return self._length

    def __getitem__(self, i):

        z = self.datas[ i % self.num_images ][0].squeeze(0)
        clip_img = self.datas[ i % self.num_images ][1].squeeze(0)
        text = self.datas[ i % self.num_images ][2].squeeze(0)
        data_type = self.datas[ i % self.num_images ][3]
        
        return z,clip_img,text,data_type