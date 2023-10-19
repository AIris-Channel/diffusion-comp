import os, re
import tqdm
import libs.autoencoder
import torch
import datetime
from PIL import Image, ImageOps
from libs.data import vae_transform
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class VAE_Dataset(Dataset):
    def __init__(self, data_root, resolution, crop_face = False):
        self.data = []
        image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root) if re.search(r'\.(?:jpe?g|png)$',file_path)]
        for image_path in image_paths:
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            image = vae_transform(resolution, crop_face=crop_face)(image)
            self.data.append(image)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def encode_decode(in_path, out_path, autoencoder, device):
    image = ImageOps.exif_transpose(Image.open(in_path)).convert("RGB")
    image = vae_transform(512,crop_face=True)(image).to(device).unsqueeze(0)
    z = autoencoder.encode(image)

    def unpreprocess(v):
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v
    
    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    z = (unpreprocess(decode(z))[0]*255).cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
    
    Image.fromarray(z).save(out_path)


def get_args():
    parser = argparse.ArgumentParser()
    # key args
    parser.add_argument('-d', '--data', type=str,
                        default="train_data/boy1", help="datadir")
    parser.add_argument('-o', "--outdir", type=str,
                        default="model_ouput/boy1", help="output of model")

    # args of logging
    parser.add_argument("--logdir", type=str, default="logs",
                        help="the dir to put logs")

    return parser.parse_args()


def train(config):
    log_interval=100
    eval_interval=100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = VAE_Dataset(config.data, resolution=512, crop_face=True)
    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    autoencoder.requires_grad_(True)
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    total_steps = 1000
    total_loss = 0
    best_score = 0
    for step in tqdm.tqdm(range(total_steps)):
        for batch in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = autoencoder.encode(batch)
            x = autoencoder.decode(z)
            loss = loss_fn(x, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if step % log_interval == 0:
            print(f"step {step}: loss {total_loss / log_interval}")
            total_loss = 0
        if step % eval_interval == 0:
            for task in ['boy1', 'boy2', 'girl1', 'girl2']:
                if task in config.data:
                    continue
            ori_imgs=[f for f in os.listdir(f'train_data/{task}') if f.endswith('jpg') or f.endswith('jpeg')]
            os.makedirs(f'pseudo/{task}_sim', exist_ok=True)
            for i in range(3):
                encode_decode(f'train_data/{task}/{random.choice(ori_imgs)}', f'pseudo/{task}_sim/0-{str(i).zfill(3)}.jpg', autoencoder, device)
            os.makedirs(f'pseudo/{task}_edit', exist_ok=True)
            for j in range(24):
                for i in range(3):
                    encode_decode(f'train_data/{task}/{random.choice(ori_imgs)}', f'pseudo/{task}_edit/{j}-{str(i).zfill(3)}.jpg', autoencoder, device)
            from score import score_one_task
            scores = score_one_task('./train_data/', './eval_prompts_advance/', './pseudo/', task)
            current_score = sum([
                scores['sim_face'] * config.sim_face_ratio +
                scores['sim_clip'] * config.sim_clip_ratio +
                scores['edit_face'] * config.edit_face_ratio +
                scores['edit_clip'] * config.edit_clip_ratio +
                scores['edit_text_clip'] * config.edit_text_clip_ratio
            ])
            if current_score > best_score:
                best_score = current_score
                print(f"step {step}: best score {best_score}")
                torch.save(autoencoder.state_dict(), os.path.join(config.outdir, 'autoencoder.pth'))
                with open('vae_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"step {step}: best score {best_score}\n")


def main():
    # 赛手需要根据自己的需求修改config file
    from configs.unidiffuserv1 import get_config
    config = get_config()
    config_name = "unidiffuserv1"
    args = get_args()
    config.log_dir = args.logdir
    config.outdir = args.outdir
    config.data = args.data
    data_name = Path(config.data).stem

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    config.workdir = os.path.join(
        config.log_dir, f"{config_name}-{data_name}-{now}")
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.meta_dir = os.path.join(config.workdir, "meta")
    os.makedirs(config.workdir, exist_ok=True)

    train(config)


if __name__ == "__main__":
    main()
