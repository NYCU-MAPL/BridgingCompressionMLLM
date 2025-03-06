import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from collections import OrderedDict

from compressai.layers import (
    AttentionBlock,
    conv3x3,
    CheckboardMaskedConv2d,
)

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x) 

class Linear_Encoder(nn.Module):
    def __init__(self, in_features, out_features, num_tokens = 1, input_resolution = 256, args = None) -> None:
        super().__init__()
        
        self.args = args
        self.num_tokens = num_tokens

        ViT_dim = 1024
        self.img_tokens_num = 49
        scale = ViT_dim ** -0.5
        self.trans_latent_dim = nn.Linear(in_features, ViT_dim)


        self.class_embedding = nn.Parameter(scale * torch.randn(ViT_dim))
        self.ln_pre  = LayerNorm(ViT_dim)

        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // 16) ** 2 + 1, ViT_dim))
        self.transformer = Transformer(ViT_dim, 1, ViT_dim//64)

    def forward(self, x):
        '''
        x: [B, in_features, H, W]
        '''
        B, C, H, W = x.shape

        x = x.view(B, C, H*W).permute(0, 2, 1) # [B, H*W, in_features]
        x = self.trans_latent_dim(x)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [B, N + 1, in_features]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [N + 1, B, in_features]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, N + 1, in_features]

        return x
