import torch

class DimTrans(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    
class ImageProjModel(torch.nn.Module):

    def __init__(self, t_dim, s_dim1, s_dim2, n_tokens):
        super().__init__()

        self.dim_trans = DimTrans(t_dim, s_dim1 + s_dim2, n_tokens)

    def forward(self, emb1, emb2):
        emb = torch.cat([emb1, emb2], dim=1)
        emb = self.dim_trans(emb)
        return emb