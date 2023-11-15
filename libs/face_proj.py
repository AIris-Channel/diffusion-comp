import torch
import torch.nn as nn
from libs.uvit_multi_post_ln_v1 import PatchEmbed

class ImageProjModel(nn.Module):
    
    def __init__(self, in_chans, patch_size, embed_dim=768, depth=30):
        super().__init__()

        self.patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            for _ in range(depth // 2 * 2 + 1)])
        
    def forward(self, image):
        outs = []
        for blk in self.patch_embeds:
            outs.append(blk(image))
        return outs

if __name__ == '__main__':
    x = torch.rand((2, 4, 116, 101))
    model = ImageProjModel(
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30
        )
    y = model(x)
