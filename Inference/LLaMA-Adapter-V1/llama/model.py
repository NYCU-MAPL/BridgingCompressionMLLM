# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import clip
import llama.CLIP_modify.clip as clip_modify

from timm.models.vision_transformer import Block

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    adapter_len: int = 10
    adapter_layer: int = 30

    cap_adapter_len: int = 10
    cap_adapter_layer: int = 30
    cap_vision_model: str = "ViT-L/14"
    cap_vision_dim: int = 512
    cap_vision_block: int = 2


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.gate = torch.nn.Parameter(torch.zeros(1))

        self.cap_gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, mode='instruct'):
        if mode == 'instruct':
            return self.forward_instruct(x, start_pos, freqs_cis, mask, adapter)
        elif mode == 'caption':
            return self.forward_caption(x, start_pos, freqs_cis, mask, adapter)

    
    def forward_instruct(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        if adapter is not None:
           adapter_len = adapter.shape[1]
           adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
           adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
           adapter_k = adapter_k.transpose(1, 2)
           adapter_v = adapter_v.transpose(1, 2)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        if adapter is not None:
            adapter_scores = torch.matmul(xq, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            adapter_scores = self.gate * F.softmax(adapter_scores.float(), dim=-1).type_as(xq)
            output = output + torch.matmul(adapter_scores, adapter_v)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)
    

    def forward_caption(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        if adapter is not None:
           adapter_len = adapter.shape[1]
           adapter_k = self.wk(adapter).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
           adapter_v = self.wv(adapter).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
           adapter_k = adapter_k.transpose(1, 2)
           adapter_v = adapter_v.transpose(1, 2)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        if adapter is not None:
            adapter_scores = torch.matmul(xq, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            adapter_scores = self.cap_gate.tanh() * F.softmax(adapter_scores.float(), dim=-1).type_as(xq)

            output = output + torch.matmul(adapter_scores, adapter_v)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)



class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, mode='instruct'):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, mode=mode)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        # Note: this is only a preview of multimodal LLaMA-Adapter
        # and requires more efforts to decouple LLaMA-Adapter from LLaMA.
        # instruct model
        self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        # caption model
        self.cap_adapter_query = nn.Embedding(params.cap_adapter_len * params.cap_adapter_layer, params.dim)
        self.cap_adapter_len = params.cap_adapter_len
        self.cap_adapter_layer = params.cap_adapter_layer

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, visual_tokens: torch.Tensor = None, mode: str = 'instruct'):
        if mode == 'instruct':
            return self.forward_instruct(tokens, start_pos, mode)
        elif mode == 'caption':
            return self.forward_caption(tokens, start_pos, visual_tokens, mode)
    
    def forward_instruct(self, tokens: torch.Tensor, start_pos: int, mode=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        adapter = self.adapter_query.weight.reshape(self.params.adapter_layer, self.params.adapter_len, self.params.dim).unsqueeze(1)
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers[: -1 * self.params.adapter_layer]:
            h = layer(h, start_pos, freqs_cis, mask)
        layer_index = 0
        for layer in self.layers[-1 * self.params.adapter_layer:]:
            h = layer(h, start_pos, freqs_cis, mask, adapter[layer_index], mode=mode)
            layer_index = layer_index + 1
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    def forward_caption(self, tokens: torch.Tensor, start_pos: int, visual_tokens: torch.Tensor = None, mode=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        adapter = self.cap_adapter_query.weight.reshape(self.params.cap_adapter_layer, self.params.cap_adapter_len, self.params.dim).unsqueeze(1)
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers[: -1 * self.params.cap_adapter_layer]:
            h = layer(h, start_pos, freqs_cis, mask)
        layer_index = 0
        for layer in self.layers[-1 * self.params.cap_adapter_layer:]:
            adapter_per_layer = adapter[layer_index]
            if visual_tokens is not None:
                adapter_per_layer = adapter_per_layer + visual_tokens
            h = layer(h, start_pos, freqs_cis, mask, adapter_per_layer, mode=mode)
            layer_index = layer_index + 1
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()



class VisionModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params

        # self.clip, self.clip_transform = clip.load(params.cap_vision_model)
        self.clip_model, self.clip_transform = clip_modify.load(params.cap_vision_model)
        self.clip_model.float()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_proj = nn.Linear(self.clip_model.visual.output_dim, params.cap_vision_dim)
        self.clip_proj_norm = nn.LayerNorm(params.cap_vision_dim)

        self.visual_query = nn.Embedding(params.cap_adapter_len, params.cap_vision_dim)

        self.visual_blocks = nn.ModuleList([
            Block(params.cap_vision_dim, 16, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
        for i in range(params.cap_vision_block)])
        
        self.visual_proj = nn.Linear(params.cap_vision_dim, params.dim)
        self.visual_proj_norm = nn.LayerNorm(params.dim)

        
        
    
    def forward(self, imgs, start_layer):
        
        _bsz = imgs.shape[0]

        _, visual_feats = self.clip_model.encode_image(image = imgs, start_layer = start_layer) 
        visual_feats = self.clip_proj_norm(self.clip_proj(visual_feats.half()))

        visual_query = self.visual_query.weight.unsqueeze(0).repeat(_bsz, 1, 1)
        visual_query = torch.cat([visual_query, visual_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)
        visual_query = visual_query[:, :self.params.cap_adapter_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)
    
        
        return visual_query