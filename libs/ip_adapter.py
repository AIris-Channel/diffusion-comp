import torch
import torch.nn as nn
import torch.nn.init as init
import einops

if hasattr(nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'


class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    

class IPAttnProcessor(nn.Module):
    def __init__(self, dim, num_heads=8, kv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qk_scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)

    def forward(self, q, k, v, B, L, C, ip_tokens):
        kv = self.kv(ip_tokens)
        q = q.float()
        if ATTENTION_MODE == 'flash':
            kv = einops.rearrange(kv, 'B L (K H D) -> K B H L D', K=2, H=self.num_heads).float()
            _k, _v = kv[0], kv[1]  # B H L D
            k = torch.cat([k, _k], dim=1)
            v = torch.cat([v, _v], dim=1)
            x = nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            kv = einops.rearrange(kv, 'B L (K H D) -> K B L H D', K=2, H=self.num_heads).float()
            _k, _v = kv[0], kv[1]  # B L H D
            k = torch.cat([k, _k], dim=1)
            v = torch.cat([v, _v], dim=1)
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            with torch.amp.autocast(device_type='cuda', enabled=False):
                kv = einops.rearrange(kv, 'B L (K H D) -> K B H L D', K=2, H=self.num_heads).float()
                _k, _v = kv[0], kv[1]  # B H L D
                k = torch.cat([k, _k], dim=1)
                v = torch.cat([v, _v], dim=1)
                _attn = (q @ k.transpose(-2, -1)) * self.qk_scale
                _attn = _attn.softmax(dim=-1)
                _attn = self.attn_drop(_attn)
                x = (_attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        return x
    
    def apply_to(self, attn):
        self.org_attn = attn.get_attn
        attn.get_attn = self.forward


class IPAdapter(nn.Module):
    def __init__(self, image_proj_model, adapter_modules):
        super().__init__()
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
    
    def forward(self, image):
        return self.image_proj_model(image)
