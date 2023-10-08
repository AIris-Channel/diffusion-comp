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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = VAE_Dataset(config.data, resolution=512, crop_face=True)
    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    autoencoder.requires_grad_(True)
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    total_steps = 1000
    for step in tqdm.tqdm(range(total_steps)):
        for batch in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = autoencoder.encode(batch)
            x = autoencoder.decode(z)
            loss = loss_fn(x, batch)
            loss.backward()
            optimizer.step()
        if step % 100 == 0:
            print(f"step: {step}, loss: {loss.item()}")
    torch.save(autoencoder.state_dict(), os.path.join(config.outdir, 'autoencoder.pth'))


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
