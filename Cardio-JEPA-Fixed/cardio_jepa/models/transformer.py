"""Transformer primitives.

Adapted from Kim's ECG-JEPA (https://github.com/sehunfromdaegu/ECG_JEPA,
`ecg_jepa.py`), which itself derives from facebookresearch/ijepa's
`src/models/vision_transformer.py`.

Changes vs. the upstream ECG-JEPA code:
  * `Attention` accepts an additive/boolean attention mask and fills with
    `torch.finfo(dtype).min` instead of `-inf`, which is NaN-safe under AMP.
  * `Block.forward` takes an optional key-padding-free mask of shape
    (N, N) or (B, 1, N, N).
  * `DropPath` is implemented locally so the package has no `timm` dependency
    (timm is not installed in the default Kaggle image).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# init helpers (cut & paste from PyTorch / timm, as in the upstream repos)
# --------------------------------------------------------------------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            # NaN-safe masking under fp16 autocast: use dtype min rather than -inf.
            neg = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(attention_mask == 0, neg)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path_rate=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, attention_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attention_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    """Queries attend to an external memory sequence."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, memory):
        B, N, C = x.shape
        M = memory.size(1)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = (self.kv(memory)
              .reshape(B, M, 2, self.num_heads, C // self.num_heads)
              .permute(2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: self-attention over queries + cross-attention to context.

    Used by the forward predictor. Keeping the context as a cross-attention memory
    (rather than concatenating it into one long sequence) keeps the attention matrix
    at O(N_query x N_ctx) instead of O((N_query + N_ctx)^2), which matters at
    C*P = 400 tokens per window on a 16 GB Kaggle GPU.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path_rate=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm_q = norm_layer(dim)
        self.norm_m = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         qk_scale=qk_scale, attn_drop=attn_drop,
                                         proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, memory, attention_mask=None):
        x = x + self.drop_path(self.self_attn(self.norm1(x), attention_mask))
        x = x + self.drop_path(self.cross_attn(self.norm_q(x), self.norm_m(memory)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderStack(nn.Module):
    def __init__(self, embed_dim=192, depth=4, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = (drop_path_rate if isinstance(drop_path_rate, list)
               else [x.item() for x in torch.linspace(0, drop_path_rate, depth)])
        self.blocks = nn.ModuleList([
            DecoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path_rate=dpr[i],
                         norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x, memory, pos=None, attention_mask=None):
        for block in self.blocks:
            block_in = x if pos is None else x + pos
            x = block(block_in, memory, attention_mask)
        return x


class TransformerStack(nn.Module):
    """A stack of Blocks that adds the positional embedding at *every* layer.

    Re-adding `pos` inside each block is the upstream ECG-JEPA behaviour
    (`Encoder_Block.forward`), kept for fidelity with the published baseline.
    """

    def __init__(self, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = (drop_path_rate if isinstance(drop_path_rate, list)
               else [x.item() for x in torch.linspace(0, drop_path_rate, depth)])
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path_rate=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x, pos=None, attention_mask=None):
        for block in self.blocks:
            block_in = x if pos is None else x + pos
            x = block(block_in, attention_mask)
        return x


def init_weights_(module: nn.Module, init_std: float = 0.02):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv1d):
        trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def rescale_blocks_(blocks: nn.ModuleList):
    """Layer-id dependent rescaling of residual projections (BEiT / ECG-JEPA trick).

    Handles both `Block` (attr `attn`) and `DecoderBlock` (attrs `self_attn`,
    `cross_attn`).
    """
    for layer_id, layer in enumerate(blocks):
        scale = math.sqrt(2.0 * (layer_id + 1))
        for name in ("attn", "self_attn", "cross_attn"):
            sub = getattr(layer, name, None)
            if sub is not None:
                sub.proj.weight.data.div_(scale)
        layer.mlp.fc2.weight.data.div_(scale)
