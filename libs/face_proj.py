import torch
import torch.nn as nn
from libs.uvit_multi_post_ln_v1 import PatchEmbed, Block

class ImageProjModel(nn.Module):
    
    def __init__(self, in_chans, patch_size, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()

        self.patch_embeds = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, skip=True, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])
        
    def forward(self, x, image, idx, skip=None):
        batch_size = x.shape[0]
        face_emb = self.patch_embeds(image)
        zero_pad = torch.zeros((batch_size, 81, 1536)).to(x.device)
        x = x + torch.cat([zero_pad, face_emb], dim=1)
        if idx < len(self.in_blocks):
            x = self.in_blocks[idx](x)
        elif idx == len(self.in_blocks):
            x = self.mid_block(x)
        else:
            x = self.out_blocks[idx - len(self.in_blocks) - 1](x, skip)
        return x

if __name__ == '__main__':
    x = torch.rand((2, 4, 116, 101))
    model = ImageProjModel(
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30
        )
    y = model(x)
