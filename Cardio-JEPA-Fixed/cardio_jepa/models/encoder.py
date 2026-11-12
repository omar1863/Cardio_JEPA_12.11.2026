"""CroPA context/target encoder.

This is Kim's ECG-JEPA `MaskTransformer` (Cross-Pattern Attention over a
lead x time patch grid) refactored into an explicit API:

    tokens = encoder(x_win, time_mask=None)     # (B, n_visible_tokens, D)
    z      = encoder.representation(x_win)      # (B, D)  mean-pooled, no masking

Cross-Pattern Attention (CroPA): a patch may attend only to patches that share
its lead (temporal context) or share its time column (cross-lead context). This
is the structural prior from arXiv:2410.08559 -- single-lead morphology plus
inter-lead physics -- and it cuts attention cost relative to dense attention over
the full C x P grid.

Masking is *synchronised across leads* (a boolean mask over the P time patches is
tiled over all C leads), exactly as in the upstream implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .pos_encoding import get_2d_sincos_pos_embed
from .transformer import TransformerStack, init_weights_, rescale_blocks_, trunc_normal_


def make_block_mask(n_patches: int, mask_scale: Tuple[float, float],
                    n_blocks: int = 4, generator: Optional[torch.Generator] = None
                    ) -> torch.Tensor:
    """Contiguous multi-block temporal mask (ECG-JEPA `_make_block_mask`)."""
    mask = torch.zeros(n_patches, dtype=torch.bool)
    for _ in range(n_blocks):
        ratio = mask_scale[0] + (mask_scale[1] - mask_scale[0]) * torch.rand(
            1, generator=generator).item()
        num = max(1, int(n_patches * ratio))
        num = min(num, n_patches - 1)
        start = torch.randint(0, n_patches - num, (1,), generator=generator).item()
        mask[start:start + num] = True
    # never mask everything -- the context encoder needs at least one visible patch
    if bool(mask.all()):
        mask[0] = False
    return mask


def make_random_mask(n_patches: int, mask_scale: Tuple[float, float],
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """i.i.d. random temporal mask (ECG-JEPA `_make_rand_mask`)."""
    mask = torch.zeros(n_patches, dtype=torch.bool)
    ratio = mask_scale[0] + (mask_scale[1] - mask_scale[0]) * torch.rand(
        1, generator=generator).item()
    num = max(1, min(int(n_patches * ratio), n_patches - 1))
    idx = torch.randperm(n_patches, generator=generator)[:num]
    mask[idx] = True
    return mask


class CroPAEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int = 384,
                 depth: int = 12,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale=None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer=nn.LayerNorm,
                 init_std: float = 0.02,
                 pos_type: str = "sincos",
                 use_cropa: bool = True,
                 n_leads: int = 8,
                 n_time_patches: int = 50,
                 patch_samples: int = 50):
        super().__init__()
        self.c = n_leads
        self.p = n_time_patches
        self.t = patch_samples
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.use_cropa = use_cropa

        self.patch_embed = nn.Linear(patch_samples, embed_dim)
        self.blocks = TransformerStack(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer)
        self.norm = norm_layer(embed_dim)

        if pos_type == "learnable":
            pos = torch.empty(self.c * self.p, embed_dim)
            nn.init.uniform_(pos, -0.02, 0.02)
            self.pos_embed = nn.Parameter(pos, requires_grad=True)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(self.c * self.p, embed_dim),
                                          requires_grad=False)
            sincos = get_2d_sincos_pos_embed(embed_dim, self.c, self.p)
            self.pos_embed.data.copy_(torch.from_numpy(sincos).float())

        # CroPA attention mask is constant -> register as a buffer (not a parameter)
        self.register_buffer("cropa_mask", self._build_cropa_mask(), persistent=False)

        self.apply(lambda m: init_weights_(m, self.init_std))
        rescale_blocks_(self.blocks.blocks)

    # ------------------------------------------------------------------ masks
    def _build_cropa_mask(self) -> torch.Tensor:
        """(C*P, C*P) binary mask: 1 where attention is allowed."""
        size = self.c * self.p
        m = torch.zeros(size, size)
        for i in range(self.c):                       # same lead  (temporal context)
            m[i * self.p:(i + 1) * self.p, i * self.p:(i + 1) * self.p] = 1
        for j in range(self.p):                       # same time column (cross-lead)
            m[j::self.p, j::self.p] = 1
        return m.clamp(max=1)

    def attention_mask(self, keep: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if not self.use_cropa:
            return None
        m = self.cropa_mask
        if keep is not None:
            m = m[keep][:, keep]
        return m

    # --------------------------------------------------------------- forward
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C*P, patch_samples)."""
        B, C, T = x.shape
        assert C == self.c, f"expected {self.c} leads, got {C}"
        assert T == self.p * self.t, f"expected {self.p * self.t} samples, got {T}"
        return x.reshape(B, self.c * self.p, self.t)

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Encode a window.

        Args:
            x: (B, C, T) raw window.
            time_mask: (P,) bool over time patches. True = masked (hidden from the
                context encoder). If None the full grid is encoded.
        Returns:
            (B, N, D) where N = C*P if `time_mask` is None else C*(P - n_masked).
        """
        x = self.tokenize(x)
        pos = self.pos_embed.unsqueeze(0).expand(x.size(0), -1, -1)
        attn = self.attention_mask()

        if time_mask is not None:
            # masks are sampled on CPU; indexing a CUDA tensor with a CPU boolean mask
            # raises "indices should be either on cpu or on the same device"
            time_mask = time_mask.to(x.device)
            keep = (~time_mask).repeat(self.c)        # lead-major tiling
            x = x[:, keep]
            pos = pos[:, keep]
            attn = self.attention_mask(keep)

        x = self.patch_embed(x)
        x = self.blocks(x, pos, attn)
        return self.norm(x)

    def full_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the full grid: (B, C, T) -> (B, C*P, D)."""
        return self.forward(x, time_mask=None)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled embedding used by every downstream probe: (B, C, T) -> (B, D)."""
        return self.full_tokens(x).mean(dim=1)


class MaskedPredictor(nn.Module):
    """Predicts masked target latents from visible context latents.

    Port of ECG-JEPA's `MaskTransformerPredictor`. The predictor runs *per lead*
    (the sequence is reshaped to (B*C, P, D)) with a shared 1-D positional
    embedding over time patches and a learned mask token, which is the
    "lead-wise predictor" described in arXiv:2410.08559.
    """

    def __init__(self,
                 embed_dim: int = 384,
                 predictor_embed_dim: int = 192,
                 depth: int = 4,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale=None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer=nn.LayerNorm,
                 init_std: float = 0.02,
                 pos_type: str = "sincos",
                 n_leads: int = 8,
                 n_time_patches: int = 50):
        super().__init__()
        self.c = n_leads
        self.p = n_time_patches
        self.embed_dim = embed_dim
        self.init_std = init_std

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        if pos_type == "learnable":
            pos = torch.empty(self.p, predictor_embed_dim)
            nn.init.uniform_(pos, -0.02, 0.02)
            self.pos_embed = nn.Parameter(pos, requires_grad=True)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(self.p, predictor_embed_dim),
                                          requires_grad=False)
            sincos = get_2d_sincos_pos_embed(predictor_embed_dim, 1, self.p)
            self.pos_embed.data.copy_(torch.from_numpy(sincos).float())

        self.blocks = TransformerStack(
            embed_dim=predictor_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        self.norm = norm_layer(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        trunc_normal_(self.mask_token, std=init_std)
        self.apply(lambda m: init_weights_(m, self.init_std))
        rescale_blocks_(self.blocks.blocks)

    def forward(self, ctx_tokens: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx_tokens: (B, C*P_vis, D) output of the context encoder.
            time_mask:  (P,) bool, True = masked.
        Returns:
            (B, C*n_masked, D) predictions for the masked patches, in lead-major order.
        """
        n_mask = int(time_mask.sum().item())
        B = ctx_tokens.size(0)

        x = self.predictor_embed(ctx_tokens)
        pdim = x.size(-1)
        x = x.reshape(B * self.c, -1, pdim)                    # (B*C, P_vis, D')
        mask_tok = self.mask_token.expand(x.size(0), n_mask, pdim)
        x = torch.cat([x, mask_tok], dim=1)                    # (B*C, P_vis+n_mask, D')

        # positional embeddings reordered to [visible..., masked...]
        vis_idx = (~time_mask).nonzero(as_tuple=True)[0]
        msk_idx = time_mask.nonzero(as_tuple=True)[0]
        order = torch.cat([vis_idx, msk_idx]).to(self.pos_embed.device)
        pos = self.pos_embed[order].unsqueeze(0).expand(x.size(0), -1, -1)

        x = self.blocks(x, pos)
        x = self.proj(self.norm(x))
        x = x.reshape(B, -1, self.embed_dim)                   # (B, C*(P_vis+n_mask), D)

        # keep only the predicted (masked) slots, per lead
        x = x.reshape(B, self.c, -1, self.embed_dim)[:, :, -n_mask:, :]
        return x.reshape(B, -1, self.embed_dim)
