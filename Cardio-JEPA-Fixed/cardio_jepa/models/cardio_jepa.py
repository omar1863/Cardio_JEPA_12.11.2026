"""Cardio-JEPA: dual-objective (masked infilling + forward latent prediction) JEPA.

    L = w_fwd * L_fwd  +  lambda * L_mask  +  beta * L_reg

  * ``L_mask`` -- Kim's masked cross-lead/cross-time infilling (morphology term).
    Reproduces arXiv:2410.08559 exactly when ``w_fwd = 0``, which is the
    masked-infilling baseline arm.
  * ``L_fwd``  -- forward latent prediction (dynamics term). Predicts the target
    encoder's latents for a *future* window at a queryable horizon. This is the
    component no published ECG-JEPA has.
  * ``L_reg``  -- the RQ3 anti-collapse regulariser.

Optionally a Level-2 "rhythm" stack (Section 5.4) forecasts future *window*
latents over a minutes-long coarse sequence built from Level-1 window summaries.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (HORIZON_BUCKETS_S, ForwardSamplingConfig, HierarchicalConfig,
                      LossConfig, ModelConfig, SignalConfig)
from ..regularizers import build_regularizer
from .encoder import CroPAEncoder, MaskedPredictor, make_block_mask, make_random_mask
from .forward_predictor import ForwardPredictor, SequencePredictor, horizon_to_bucket
from .pos_encoding import get_1d_sincos_pos_embed
from .transformer import TransformerStack, init_weights_, rescale_blocks_


def prediction_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "smooth_l1"
                    ) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "l2":
        return F.mse_loss(pred, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"unknown loss_fn: {kind!r}")


def normalize_target(h: torch.Tensor) -> torch.Tensor:
    """LayerNorm over the feature dimension, as in I-JEPA / ECG-JEPA target processing."""
    return F.layer_norm(h.float(), (h.size(-1),)).to(h.dtype)


class Level2Stack(nn.Module):
    """Rhythm-scale (Level-2) context encoder + EMA target + forward predictor."""

    def __init__(self, embed_dim: int, cfg: HierarchicalConfig, n_horizon_buckets: int,
                 max_context_len: int = 64, n_mask_slots: int = 2,
                 use_ema_target: bool = True):
        super().__init__()
        self.cfg = cfg
        self.n_mask_slots = n_mask_slots
        self.use_ema_target = use_ema_target

        self.in_proj = nn.Linear(embed_dim, cfg.l2_embed_dim)
        self.encoder = TransformerStack(
            embed_dim=cfg.l2_embed_dim, depth=cfg.l2_depth, num_heads=cfg.l2_num_heads,
            qkv_bias=True)
        self.norm = nn.LayerNorm(cfg.l2_embed_dim)
        self.register_buffer(
            "pos", torch.from_numpy(
                get_1d_sincos_pos_embed(cfg.l2_embed_dim, max_context_len)).float(),
            persistent=False)

        self.fwd_predictor = SequencePredictor(
            embed_dim=cfg.l2_embed_dim, predictor_embed_dim=cfg.l2_predictor_dim,
            depth=cfg.l2_predictor_depth, num_heads=cfg.l2_num_heads,
            max_context_len=max_context_len, n_query=1,
            n_horizon_buckets=n_horizon_buckets)
        self.mask_predictor = SequencePredictor(
            embed_dim=cfg.l2_embed_dim, predictor_embed_dim=cfg.l2_predictor_dim,
            depth=cfg.l2_predictor_depth, num_heads=cfg.l2_num_heads,
            max_context_len=max_context_len, n_query=n_mask_slots,
            n_horizon_buckets=n_horizon_buckets)

        self.apply(lambda m: init_weights_(m, 0.02))
        rescale_blocks_(self.encoder.blocks)

        if use_ema_target:
            self.target_in_proj = copy.deepcopy(self.in_proj)
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_norm = copy.deepcopy(self.norm)
            for p in self.target_parameters():
                p.requires_grad = False
        else:
            self.target_in_proj = None
            self.target_encoder = None
            self.target_norm = None

    def target_parameters(self):
        if self.target_encoder is None:
            return []
        return (list(self.target_in_proj.parameters())
                + list(self.target_encoder.parameters())
                + list(self.target_norm.parameters()))

    def online_parameters(self):
        return (list(self.in_proj.parameters())
                + list(self.encoder.parameters())
                + list(self.norm.parameters()))

    def encode(self, summaries: torch.Tensor, target: bool = False,
               keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, L, D_l1) -> (B, L, D_l2)."""
        in_proj = self.target_in_proj if (target and self.target_in_proj is not None) else self.in_proj
        enc = self.target_encoder if (target and self.target_encoder is not None) else self.encoder
        norm = self.target_norm if (target and self.target_norm is not None) else self.norm

        x = in_proj(summaries)
        pos = self.pos[:summaries.size(1)].unsqueeze(0)
        if keep is not None:
            x = x[:, keep]
            pos = pos[:, keep]
        return norm(enc(x, pos))


class CardioJEPA(nn.Module):
    def __init__(self,
                 signal: SignalConfig,
                 model: ModelConfig,
                 loss: LossConfig,
                 hierarchical: Optional[HierarchicalConfig] = None,
                 forward_sampling: Optional[ForwardSamplingConfig] = None,
                 horizon_buckets: Sequence[float] = HORIZON_BUCKETS_S,
                 max_context_windows: int = 16,
                 reg_tokens: int = 32):
        super().__init__()
        self.reg_tokens = reg_tokens
        self.reg_target = getattr(loss, "reg_target", "pooled")
        self.signal = signal
        self.model_cfg = model
        self.loss_cfg = loss
        self.hier_cfg = hierarchical or HierarchicalConfig()
        self.fwd_cfg = forward_sampling or ForwardSamplingConfig()
        self.horizon_buckets = tuple(horizon_buckets)
        self.max_context_windows = max_context_windows
        self.use_ema_target = loss.use_ema_target

        C, P, t = signal.n_leads, signal.n_time_patches, signal.patch_samples
        D = model.encoder_embed_dim

        self.encoder = CroPAEncoder(
            embed_dim=D, depth=model.encoder_depth, num_heads=model.encoder_num_heads,
            mlp_ratio=model.mlp_ratio, qkv_bias=model.qkv_bias,
            drop_rate=model.drop_rate, attn_drop_rate=model.attn_drop_rate,
            drop_path_rate=model.drop_path_rate, init_std=model.init_std,
            pos_type=model.pos_type, use_cropa=model.use_cropa,
            n_leads=C, n_time_patches=P, patch_samples=t)

        if self.use_ema_target:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            # SIGReg / LeJEPA arm: no second network, targets are stop-gradiented
            # outputs of the online encoder itself.
            self.target_encoder = None

        self.masked_predictor = MaskedPredictor(
            embed_dim=D, predictor_embed_dim=model.predictor_embed_dim,
            depth=model.predictor_depth, num_heads=model.predictor_num_heads,
            mlp_ratio=model.mlp_ratio, qkv_bias=model.qkv_bias,
            drop_rate=model.drop_rate, attn_drop_rate=model.attn_drop_rate,
            drop_path_rate=model.drop_path_rate, init_std=model.init_std,
            pos_type=model.pos_type, n_leads=C, n_time_patches=P)

        self.forward_predictor = ForwardPredictor(
            embed_dim=D, predictor_embed_dim=model.predictor_embed_dim,
            depth=model.predictor_depth, num_heads=model.predictor_num_heads,
            mlp_ratio=model.mlp_ratio, qkv_bias=model.qkv_bias,
            drop_rate=model.drop_rate, attn_drop_rate=model.attn_drop_rate,
            drop_path_rate=model.drop_path_rate, init_std=model.init_std,
            n_leads=C, n_time_patches=P, n_horizon_buckets=len(self.horizon_buckets),
            max_context_windows=max_context_windows,
            output_mode=model.forward_target,
            use_cropa_queries=model.predictor_cropa)

        self.regularizer = build_regularizer(loss, D)

        self.level2_regularizer = None
        self.level2 = None
        if self.hier_cfg.enabled:
            self.level2_regularizer = build_regularizer(loss, self.hier_cfg.l2_embed_dim)
            self.level2 = Level2Stack(
                embed_dim=D, cfg=self.hier_cfg,
                n_horizon_buckets=len(self.horizon_buckets),
                max_context_len=max(64, self.hier_cfg.l2_context_windows + 4),
                use_ema_target=self.use_ema_target)

    # ------------------------------------------------------------------ helpers
    @property
    def embed_dim(self) -> int:
        return self.model_cfg.encoder_embed_dim

    def _target_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Stop-gradiented, layer-normed target latents for a window batch."""
        with torch.no_grad():
            enc = self.target_encoder if self.target_encoder is not None else self.encoder
            h = enc.full_tokens(x)
        return normalize_target(h)

    def sample_time_mask(self, device=None) -> torch.Tensor:
        P = self.signal.n_time_patches
        if self.model_cfg.mask_type == "random":
            m = make_random_mask(P, self.model_cfg.mask_scale)
        else:
            m = make_block_mask(P, self.model_cfg.mask_scale)
        return m.to(device) if device is not None else m

    @torch.no_grad()
    def update_target(self, momentum: float):
        """EMA update: theta_bar <- m * theta_bar + (1 - m) * theta."""
        if self.target_encoder is not None:
            for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                k.data.mul_(momentum).add_(q.detach().data, alpha=1.0 - momentum)
            for qb, kb in zip(self.encoder.buffers(), self.target_encoder.buffers()):
                kb.data.copy_(qb.data)
        if self.level2 is not None and self.level2.target_encoder is not None:
            for q, k in zip(self.level2.online_parameters(),
                            self.level2.target_parameters()):
                k.data.mul_(momentum).add_(q.detach().data, alpha=1.0 - momentum)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder embedding used by every downstream probe: (B, C, T) -> (B, D)."""
        return self.encoder.representation(x)

    def _intra_window_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal forward prediction inside a single window.

        Context = the first ``intra_window_frac`` of the time patches, target = the
        remaining (later) patches. The encoder never sees the future patches, so this
        is genuine forecasting rather than infilling: it differs from ``L_mask`` in
        that the held-out region is always a *suffix*, and the prediction is
        conditioned on a horizon token.
        """
        B, C, T = x.shape
        P = self.signal.n_time_patches
        D = self.embed_dim
        device = x.device

        k = int(round(P * self.fwd_cfg.intra_window_frac))
        k = max(1, min(k, P - 1))

        time_mask = torch.zeros(P, dtype=torch.bool)
        time_mask[k:] = True                                   # future = hidden

        # grid indices (lead-major, matching CroPAEncoder's token ordering)
        lead = torch.arange(C).unsqueeze(1)
        ctx_idx = (lead * P + torch.arange(0, k).unsqueeze(0)).reshape(-1)
        q_idx = (lead * P + torch.arange(k, P).unsqueeze(0)).reshape(-1)

        ctx_vis = self.encoder(x, time_mask=time_mask)          # (B, C*k, D)

        h_full = self._target_tokens(x).reshape(B, C, P, D)
        h_tgt = h_full[:, :, k:, :].reshape(B, -1, D)

        # horizon = centre of the target span measured from the end of the context
        patch_s = self.signal.patch_samples / self.signal.fs
        horizon = ((P - k) / 2.0) * patch_s
        h_idx = horizon_to_bucket(torch.full((B,), horizon), self.horizon_buckets).to(device)

        if self.model_cfg.forward_target == "pooled":
            h_tgt = h_tgt.mean(dim=1)
            pred = self.forward_predictor(ctx_vis, h_idx, n_context_windows=1,
                                          ctx_pos_index=ctx_idx)
        else:
            pred = self.forward_predictor(ctx_vis, h_idx, n_context_windows=1,
                                          ctx_pos_index=ctx_idx, query_pos_index=q_idx)

        h_tgt = self.regularizer.transform_target(h_tgt)
        return prediction_loss(pred, h_tgt, self.loss_cfg.loss_fn)

    # ------------------------------------------------------------------ forward
    def forward(self,
                ctx_windows: torch.Tensor,
                future_window: Optional[torch.Tensor] = None,
                horizon_s: Optional[torch.Tensor] = None,
                l2_future: Optional[torch.Tensor] = None,
                l2_horizon_s: Optional[torch.Tensor] = None,
                use_mask_branch: bool = True,
                use_forward_branch: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            ctx_windows:   (B, W, C, T) contiguous context windows, most recent last.
            future_window: (B, C, T) target window at ``horizon_s`` after the context.
            horizon_s:     (B,) forecast horizon in seconds.
            l2_future:     (B, C, T) far-future window for the Level-2 rhythm stack.
            l2_horizon_s:  (B,) Level-2 horizon in seconds.
        Returns:
            dict with ``loss`` plus its components and ``z_ctx`` for diagnostics.
        """
        if ctx_windows.dim() == 3:
            ctx_windows = ctx_windows.unsqueeze(1)
        B, W, C, T = ctx_windows.shape
        if W > self.max_context_windows:
            raise ValueError(f"{W} context windows exceeds max_context_windows="
                             f"{self.max_context_windows}")
        device = ctx_windows.device
        D = self.embed_dim

        out: Dict[str, torch.Tensor] = {}
        zero = ctx_windows.new_zeros(())
        loss_mask = zero
        loss_fwd = zero
        loss_l2 = zero

        # ---------------- context encoding (shared by both branches) ----------------
        flat = ctx_windows.reshape(B * W, C, T)
        ctx_tokens = self.encoder.full_tokens(flat)                  # (B*W, C*P, D)
        ctx_tokens = ctx_tokens.reshape(B, W * ctx_tokens.size(1), D)

        # ---------------- (a) masked infilling: morphology --------------------------
        if use_mask_branch and self.loss_cfg.lambda_mask > 0:
            x_mask = ctx_windows[:, -1]                              # most recent window
            time_mask = self.sample_time_mask()
            h_full = self._target_tokens(x_mask)                     # (B, C*P, D)
            h_full = h_full.reshape(B, C, self.signal.n_time_patches, D)
            h_tgt = h_full[:, :, time_mask.to(h_full.device), :].reshape(B, -1, D)
            h_tgt = self.regularizer.transform_target(h_tgt)

            ctx_vis = self.encoder(x_mask, time_mask=time_mask)
            pred = self.masked_predictor(ctx_vis, time_mask)
            loss_mask = prediction_loss(pred, h_tgt, self.loss_cfg.loss_fn)

        # ---------------- (b) forward latent prediction: dynamics -------------------
        if use_forward_branch and self.loss_cfg.weight_forward > 0:
            if future_window is not None:
                # cross-window forecasting (long recordings: LTAFDB, MIT-BIH)
                if horizon_s is None:
                    horizon_s = torch.full((B,), self.horizon_buckets[0], device=device)
                h_idx = horizon_to_bucket(horizon_s.detach().cpu(),
                                          self.horizon_buckets).to(device)

                h_fut = self._target_tokens(future_window)           # (B, C*P, D)
                if self.model_cfg.forward_target == "pooled":
                    h_fut = h_fut.mean(dim=1)                        # (B, D)
                h_fut = self.regularizer.transform_target(h_fut)

                pred_fut = self.forward_predictor(ctx_tokens, h_idx, n_context_windows=W)
                loss_fwd = prediction_loss(pred_fut, h_fut, self.loss_cfg.loss_fn)

            elif self.fwd_cfg.intra_window:
                # intra-window (causal) forecasting for 10 s clips that have no
                # "next window": predict the later time patches from the earlier ones.
                loss_fwd = self._intra_window_forward(ctx_windows[:, -1])

        # ---------------- (c) Level-2 rhythm stack (RQ2) ----------------------------
        if self.level2 is not None and l2_future is not None:
            summaries = ctx_tokens.reshape(B, W, -1, D).mean(dim=2)  # (B, W, D)
            l2_ctx = self.level2.encode(summaries)                   # (B, W, D2)

            if l2_horizon_s is None:
                l2_horizon_s = torch.full((B,), self.hier_cfg.l2_horizons_s[0],
                                          device=device)
            l2_idx = horizon_to_bucket(l2_horizon_s.detach().cpu(),
                                       self.horizon_buckets).to(device)

            fut_summary = self._target_tokens(l2_future).mean(dim=1, keepdim=True)
            with torch.no_grad():
                l2_tgt = self.level2.encode(fut_summary, target=True)
            l2_tgt = normalize_target(l2_tgt)
            l2_tgt = self.level2_regularizer.transform_target(l2_tgt)

            l2_pred = self.level2.fwd_predictor(l2_ctx, l2_idx)      # (B, 1, D2)
            loss_l2 = prediction_loss(l2_pred, l2_tgt, self.loss_cfg.loss_fn)

            # Level-2 masked infilling over the coarse sequence
            if self.hier_cfg.l2_mask_weight > 0 and W > self.level2.n_mask_slots + 1:
                n_mask = self.level2.n_mask_slots
                perm = torch.randperm(W)[:n_mask].sort().values
                keep = torch.ones(W, dtype=torch.bool)
                keep[perm] = False
                l2_vis = self.level2.encode(summaries, keep=keep)
                with torch.no_grad():
                    l2_all = self.level2.encode(summaries.detach(), target=True)
                l2_masked_tgt = normalize_target(l2_all[:, perm])
                l2_masked_tgt = self.level2_regularizer.transform_target(l2_masked_tgt)
                l2_masked_pred = self.level2.mask_predictor(l2_vis, l2_idx)
                loss_l2 = loss_l2 + self.hier_cfg.l2_mask_weight * prediction_loss(
                    l2_masked_pred, l2_masked_tgt, self.loss_cfg.loss_fn)

            l2_reg, _ = self.level2_regularizer(l2_ctx.mean(1))
            loss_l2 = self.hier_cfg.l2_weight * (loss_l2 + self.loss_cfg.beta_reg * l2_reg)

        # ---------------- (d) anti-collapse regulariser (RQ3) -----------------------
        # Regularise the *pooled per-sample* representation by default.
        #
        # Applying it to the flattened token matrix (B*N, D) instead measures variance
        # across samples and patch positions mixed together -- and position alone
        # supplies plenty. Measured here: token-level batch-std 0.110 while the pooled
        # per-sample std was 0.00047, a 233x gap. The hinge was satisfied by positional
        # variance while every *sample* still mapped to the same pooled vector, which is
        # why raising beta from 0.05 to 4.0 changed the outcome by <1e-4. The pooled
        # vector is also exactly what every downstream probe consumes, so it is the
        # thing that has to stay non-degenerate.
        z_pooled = ctx_tokens.reshape(B, W, -1, D).mean(dim=2).reshape(B * W, D)

        if self.reg_target in ("tokens", "both"):
            n_tok = ctx_tokens.size(1)
            if self.reg_tokens and n_tok > self.reg_tokens:
                sel = torch.randperm(n_tok, device=ctx_tokens.device)[:self.reg_tokens]
                z_tok = ctx_tokens[:, sel].reshape(-1, D)
            else:
                z_tok = ctx_tokens.reshape(-1, D)

        if self.reg_target == "pooled":
            loss_reg, reg_stats = self.regularizer(z_pooled)
        elif self.reg_target == "tokens":
            loss_reg, reg_stats = self.regularizer(z_tok)
        else:                                            # 'both'
            lp, reg_stats = self.regularizer(z_pooled)
            lt, _ = self.regularizer(z_tok)
            loss_reg = lp + 0.5 * lt

        z_ctx = z_pooled

        total = (self.loss_cfg.weight_forward * loss_fwd
                 + self.loss_cfg.lambda_mask * loss_mask
                 + self.loss_cfg.beta_reg * loss_reg
                 + loss_l2)

        out.update({
            "loss": total,
            "loss_fwd": loss_fwd.detach(),
            "loss_mask": loss_mask.detach(),
            "loss_reg": loss_reg.detach() if torch.is_tensor(loss_reg) else loss_reg,
            "loss_l2": loss_l2.detach() if torch.is_tensor(loss_l2) else loss_l2,
            "z_ctx": z_ctx.detach(),
        })
        for k, v in reg_stats.items():
            out[k] = torch.as_tensor(v)
        return out

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def forecast_latent(self, ctx_windows: torch.Tensor, horizon_s: float) -> torch.Tensor:
        """Query the world model: predicted future latent at ``horizon_s`` seconds.

        Returns (B, D) -- mean-pooled if the predictor is in ``tokens`` mode. This is
        the feature used by the anticipation probes on top of a frozen encoder.
        """
        self.eval()
        if ctx_windows.dim() == 3:
            ctx_windows = ctx_windows.unsqueeze(1)
        B, W, C, T = ctx_windows.shape
        flat = ctx_windows.reshape(B * W, C, T)
        tok = self.encoder.full_tokens(flat)
        tok = tok.reshape(B, W * tok.size(1), self.embed_dim)
        h_idx = horizon_to_bucket(
            torch.full((B,), float(horizon_s)), self.horizon_buckets).to(ctx_windows.device)
        pred = self.forward_predictor(tok, h_idx, n_context_windows=W)
        return pred.mean(dim=1) if pred.dim() == 3 else pred

    @torch.no_grad()
    def anticipation_features(self, ctx_windows: torch.Tensor,
                              horizons_s: Sequence[float]) -> torch.Tensor:
        """[ pooled context embedding | predicted future latent per horizon ].

        This is the representation the anticipation benchmarks probe: it exposes both
        "what the heart looks like now" and "where the world model thinks it is going".
        """
        self.eval()
        if ctx_windows.dim() == 3:
            ctx_windows = ctx_windows.unsqueeze(1)
        B, W, C, T = ctx_windows.shape
        flat = ctx_windows.reshape(B * W, C, T)
        tok = self.encoder.full_tokens(flat)
        pooled = tok.reshape(B, W, -1, self.embed_dim).mean(dim=(1, 2))     # (B, D)
        feats: List[torch.Tensor] = [pooled]
        tok = tok.reshape(B, W * tok.size(1), self.embed_dim)
        for h in horizons_s:
            h_idx = horizon_to_bucket(torch.full((B,), float(h)),
                                      self.horizon_buckets).to(ctx_windows.device)
            pred = self.forward_predictor(tok, h_idx, n_context_windows=W)
            feats.append(pred.mean(dim=1) if pred.dim() == 3 else pred)
        if self.level2 is not None:
            summaries = tok.reshape(B, W, -1, self.embed_dim).mean(2)
            rhythm = self.level2.encode(summaries)
            feats.append(rhythm.mean(1))
            for h in horizons_s:
                idx = horizon_to_bucket(torch.full((B,), float(h)), self.horizon_buckets).to(tok.device)
                feats.append(self.level2.fwd_predictor(rhythm, idx).mean(1))
        return torch.cat(feats, dim=-1)
