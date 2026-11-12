"""Forward (future) latent predictor -- the novel component of Cardio-JEPA.

Section 5.3(b) of the proposal:

    z_ctx = f_theta(x_[t-L, t])            # context encoder, gradients flow
    z_fut = sg(fbar_theta(x_[t, t+H]))     # EMA target encoder, stop-gradient
    L_fwd = || g_phi(z_ctx, h) - z_fut ||_1

`h` is a learned *horizon token*, indexed by a discretised horizon bucket
({10 s, 30 s, 1 min, 5 min, 10 min, 30 min, 1 hr} by default), so a single
predictor can be queried at multiple forecast horizons -- this is what makes the
model a queryable cardiac world model rather than a fixed-lag forecaster.

Two output modes:
  * ``tokens``  -- predict the full (C*P, D) future patch grid. Richer signal,
                   what we use by default.
  * ``pooled``  -- predict a single (D,) window-level future latent. Cheaper;
                   useful for long horizons and for the Level-2 rhythm stack.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .pos_encoding import get_2d_sincos_pos_embed
from .transformer import DecoderStack, init_weights_, rescale_blocks_, trunc_normal_


def horizon_to_bucket(horizon_s, buckets: Sequence[float]) -> torch.Tensor:
    """Map horizons in seconds to the index of the nearest bucket (in log-space).

    Works on floats, lists, numpy arrays and tensors; always returns a long tensor.
    """
    h = torch.as_tensor(horizon_s, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(list(buckets), dtype=torch.float32)
    # nearest in log space, so 10 s vs 30 s is as distinguishable as 10 min vs 30 min
    d = (torch.log(h.clamp_min(1e-3)).unsqueeze(1) - torch.log(b).unsqueeze(0)).abs()
    return d.argmin(dim=1).long()


class ForwardPredictor(nn.Module):
    """g_phi: (context latents, horizon) -> predicted future latents."""

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
                 n_leads: int = 8,
                 n_time_patches: int = 50,
                 n_horizon_buckets: int = 7,
                 max_context_windows: int = 8,
                 output_mode: str = "tokens",
                 use_cropa_queries: bool = False):
        super().__init__()
        assert output_mode in ("tokens", "pooled")
        self.c = n_leads
        self.p = n_time_patches
        self.embed_dim = embed_dim
        self.output_mode = output_mode
        self.init_std = init_std
        self.use_cropa_queries = use_cropa_queries

        pdim = predictor_embed_dim
        self.ctx_proj = nn.Linear(embed_dim, pdim, bias=True)

        # --- context side: 2-D (lead, time) grid position + window-index embedding ---
        grid = get_2d_sincos_pos_embed(pdim, self.c, self.p)
        self.register_buffer("grid_pos", torch.from_numpy(grid).float(), persistent=False)
        self.window_embed = nn.Parameter(torch.zeros(max_context_windows, pdim))

        # --- horizon conditioning ---
        self.horizon_embed = nn.Embedding(n_horizon_buckets, pdim)

        # --- query side ---
        n_query = self.c * self.p if output_mode == "tokens" else 1
        self.n_query = n_query
        self.query_token = nn.Parameter(torch.zeros(1, 1, pdim))
        if output_mode == "tokens":
            self.register_buffer("query_pos", torch.from_numpy(grid).float(),
                                 persistent=False)
        else:
            self.register_buffer("query_pos", torch.zeros(1, pdim), persistent=False)

        self.blocks = DecoderStack(
            embed_dim=pdim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer)
        self.norm = norm_layer(pdim)
        self.proj = nn.Linear(pdim, embed_dim, bias=True)

        if use_cropa_queries and output_mode == "tokens":
            self.register_buffer("query_attn_mask", self._build_cropa_mask(),
                                 persistent=False)
        else:
            self.query_attn_mask = None

        trunc_normal_(self.query_token, std=init_std)
        trunc_normal_(self.window_embed, std=init_std)
        self.apply(lambda m: init_weights_(m, self.init_std))
        rescale_blocks_(self.blocks.blocks)

    def _build_cropa_mask(self) -> torch.Tensor:
        size = self.c * self.p
        m = torch.zeros(size, size)
        for i in range(self.c):
            m[i * self.p:(i + 1) * self.p, i * self.p:(i + 1) * self.p] = 1
        for j in range(self.p):
            m[j::self.p, j::self.p] = 1
        return m.clamp(max=1)

    def forward(self, ctx_tokens: torch.Tensor, horizon_idx: torch.Tensor,
                n_context_windows: int = 1,
                ctx_pos_index: Optional[torch.Tensor] = None,
                query_pos_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            ctx_tokens:  (B, n_context_windows * n_ctx, D) context-encoder output.
            horizon_idx: (B,) long tensor of horizon-bucket indices.
            n_context_windows: how many windows are stacked inside ctx_tokens.
            ctx_pos_index: (n_ctx,) indices into the C*P grid that ``ctx_tokens``
                occupy within each window. ``None`` means the full grid. This is what
                lets the *intra-window* forward objective feed only the past patches
                of a single 10 s recording as context.
            query_pos_index: (n_q,) grid indices to predict. ``None`` means the full
                grid (tokens mode) or a single pooled slot (pooled mode).
        Returns:
            ``tokens`` mode: (B, n_q, D) predicted future patch latents.
            ``pooled`` mode: (B, D)      predicted future window latent.
        """
        B = ctx_tokens.size(0)
        device = ctx_tokens.device
        mem = self.ctx_proj(ctx_tokens)

        # context positional embedding: grid position + which window it came from
        base = self.grid_pos
        if ctx_pos_index is not None:
            base = base[ctx_pos_index.to(base.device)]
        tiles = [base + self.window_embed[w] for w in range(n_context_windows)]
        ctx_pos = torch.cat(tiles, dim=0).unsqueeze(0)                 # (1, W*n_ctx, D')
        if ctx_pos.size(1) != mem.size(1):
            raise ValueError(
                f"context has {mem.size(1)} tokens but positional layout expects "
                f"{ctx_pos.size(1)} (n_context_windows={n_context_windows}, "
                f"n_ctx_per_window={base.size(0)})")
        mem = mem + ctx_pos

        # horizon token is appended to the memory so every query can attend to it
        h_tok = self.horizon_embed(horizon_idx.to(device)).unsqueeze(1)
        mem = torch.cat([mem, h_tok], dim=1)                           # (B, W*n_ctx+1, D')

        # queries: learned token + target position + the same horizon embedding,
        # so the horizon is available both additively and through attention
        if query_pos_index is None:
            q_pos = self.query_pos
            attn_mask = self.query_attn_mask
        else:
            qi = query_pos_index.to(self.query_pos.device)
            q_pos = self.query_pos[qi]
            attn_mask = (self.query_attn_mask[qi][:, qi]
                         if self.query_attn_mask is not None else None)
        n_q = q_pos.size(0)

        q = self.query_token.expand(B, n_q, -1)
        q = q + q_pos.unsqueeze(0) + h_tok

        q = self.blocks(q, mem, attention_mask=attn_mask)
        q = self.proj(self.norm(q))                                    # (B, n_q, D)

        if self.output_mode == "pooled":
            return q.squeeze(1)
        return q


class SequencePredictor(nn.Module):
    """Level-2 forward predictor over a coarse sequence of window summaries.

    Section 5.4: Level 1 works on the patch grid inside a 10 s window; Level 2
    pools per-window summary tokens into a minutes-long sequence and forecasts
    future *window* latents. Same JEPA recipe, one time-scale up.
    """

    def __init__(self,
                 embed_dim: int = 384,
                 predictor_embed_dim: int = 192,
                 depth: int = 2,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer=nn.LayerNorm,
                 init_std: float = 0.02,
                 max_context_len: int = 64,
                 n_query: int = 1,
                 n_horizon_buckets: int = 7):
        super().__init__()
        pdim = predictor_embed_dim
        self.embed_dim = embed_dim
        self.n_query = n_query
        self.init_std = init_std

        self.ctx_proj = nn.Linear(embed_dim, pdim, bias=True)
        self.horizon_embed = nn.Embedding(n_horizon_buckets, pdim)

        from .pos_encoding import get_1d_sincos_pos_embed
        self.register_buffer(
            "ctx_pos",
            torch.from_numpy(get_1d_sincos_pos_embed(pdim, max_context_len)).float(),
            persistent=False)
        self.register_buffer(
            "query_pos",
            torch.from_numpy(get_1d_sincos_pos_embed(pdim, max(n_query, 1))).float(),
            persistent=False)

        self.query_token = nn.Parameter(torch.zeros(1, 1, pdim))
        self.blocks = DecoderStack(
            embed_dim=pdim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        self.norm = norm_layer(pdim)
        self.proj = nn.Linear(pdim, embed_dim, bias=True)

        trunc_normal_(self.query_token, std=init_std)
        self.apply(lambda m: init_weights_(m, self.init_std))
        rescale_blocks_(self.blocks.blocks)

    def forward(self, ctx_seq: torch.Tensor, horizon_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx_seq: (B, L, D) sequence of Level-1 window summaries.
            horizon_idx: (B,) long.
        Returns:
            (B, n_query, D) predicted future window summaries.
        """
        B, L, _ = ctx_seq.shape
        if L > self.ctx_pos.size(0):
            raise ValueError(f"context length {L} exceeds max_context_len "
                             f"{self.ctx_pos.size(0)}")
        mem = self.ctx_proj(ctx_seq) + self.ctx_pos[:L].unsqueeze(0)
        h_tok = self.horizon_embed(horizon_idx.to(ctx_seq.device)).unsqueeze(1)
        mem = torch.cat([mem, h_tok], dim=1)

        q = self.query_token.expand(B, self.n_query, -1)
        q = q + self.query_pos[:self.n_query].unsqueeze(0) + h_tok
        q = self.blocks(q, mem)
        return self.proj(self.norm(q))
