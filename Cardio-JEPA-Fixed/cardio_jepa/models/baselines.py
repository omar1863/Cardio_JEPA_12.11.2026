"""Baselines for Section 6.3.

  * ``STMEMBaseline``  -- masked *raw-signal* reconstruction over the same
    lead x time patch grid (ST-MEM / ECG-MAE family). Same encoder, same masking,
    but the target is the raw patch instead of a latent, which is exactly the
    "wastes capacity reconstructing high-frequency noise" limitation the proposal
    argues against. Kept architecture-matched so the comparison isolates the
    objective.
  * ``XResNet1d`` / ``Inception1d`` -- supervised-from-scratch references
    (the standard PTB-XL benchmark CNNs).

The masked-infilling ECG-JEPA baseline (Kim 2024) is not a separate class: it is
``CardioJEPA`` with ``weight_forward = 0``, so the two arms share every line of
code except the objective.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CroPAEncoder, make_block_mask, make_random_mask
from .pos_encoding import get_2d_sincos_pos_embed
from .transformer import TransformerStack, init_weights_, trunc_normal_


class STMEMBaseline(nn.Module):
    """Masked raw-signal reconstruction with the same CroPA encoder."""

    def __init__(self, signal, model):
        super().__init__()
        self.signal = signal
        self.model_cfg = model
        C, P, t = signal.n_leads, signal.n_time_patches, signal.patch_samples
        D = model.encoder_embed_dim
        dec_dim = model.predictor_embed_dim

        self.encoder = CroPAEncoder(
            embed_dim=D, depth=model.encoder_depth, num_heads=model.encoder_num_heads,
            mlp_ratio=model.mlp_ratio, qkv_bias=model.qkv_bias,
            drop_path_rate=model.drop_path_rate, init_std=model.init_std,
            pos_type=model.pos_type, use_cropa=model.use_cropa,
            n_leads=C, n_time_patches=P, patch_samples=t)

        self.decoder_embed = nn.Linear(D, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer(
            "decoder_pos",
            torch.from_numpy(get_2d_sincos_pos_embed(dec_dim, C, P)).float(),
            persistent=False)
        self.decoder = TransformerStack(embed_dim=dec_dim, depth=model.predictor_depth,
                                        num_heads=model.predictor_num_heads,
                                        qkv_bias=model.qkv_bias)
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, t)

        trunc_normal_(self.mask_token, std=model.init_std)
        self.apply(lambda m: init_weights_(m, model.init_std))

    def representation(self, x):
        return self.encoder.representation(x)

    def forward(self, x: torch.Tensor):
        """x: (B, C, T) -> dict with the reconstruction loss on masked patches."""
        if x.dim() == 4:
            x = x[:, -1]
        B, C, T = x.shape
        P, t = self.signal.n_time_patches, self.signal.patch_samples
        D = self.model_cfg.encoder_embed_dim

        if self.model_cfg.mask_type == "random":
            time_mask = make_random_mask(P, self.model_cfg.mask_scale)
        else:
            time_mask = make_block_mask(P, self.model_cfg.mask_scale)
        time_mask = time_mask.to(x.device)

        vis = self.encoder(x, time_mask=time_mask)                 # (B, C*P_vis, D)
        vis = self.decoder_embed(vis)

        keep = (~time_mask).repeat(C)
        n_tokens = C * P
        full = self.mask_token.expand(B, n_tokens, vis.size(-1)).clone()
        full[:, keep] = vis.to(full.dtype)
        full = full + self.decoder_pos.unsqueeze(0)

        dec = self.decoder_pred(self.decoder_norm(self.decoder(full)))   # (B, C*P, t)

        target = x.reshape(B, C * P, t)
        sel = time_mask.repeat(C)
        # per-patch normalisation, as in MAE/ST-MEM
        tgt = target[:, sel]
        mu = tgt.mean(dim=-1, keepdim=True)
        sd = tgt.std(dim=-1, keepdim=True) + 1e-6
        loss = F.mse_loss(dec[:, sel], (tgt - mu) / sd)
        return {"loss": loss, "loss_recon": loss.detach(),
                "z_ctx": self.encoder.full_tokens(x).mean(dim=1).detach()}

    def update_target(self, momentum: float):   # API parity with CardioJEPA
        return


# --------------------------------------------------------------------------------------
# Supervised references
# --------------------------------------------------------------------------------------
def _conv(ni, nf, ks=3, stride=1):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=False)


class ResBlock1d(nn.Module):
    def __init__(self, ni, nf, stride=1, ks=5):
        super().__init__()
        self.conv1 = _conv(ni, nf, ks, stride)
        self.bn1 = nn.BatchNorm1d(nf)
        self.conv2 = _conv(nf, nf, ks, 1)
        self.bn2 = nn.BatchNorm1d(nf)
        self.act = nn.ReLU(inplace=True)
        self.short = (nn.Identity() if (ni == nf and stride == 1)
                      else nn.Sequential(nn.Conv1d(ni, nf, 1, stride=stride, bias=False),
                                         nn.BatchNorm1d(nf)))

    def forward(self, x):
        idt = self.short(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + idt)


class XResNet1d(nn.Module):
    """Compact xresnet1d-style supervised baseline (PTB-XL benchmark family)."""

    def __init__(self, n_leads: int = 8, num_classes: int = 5,
                 widths: List[int] = (64, 128, 256, 512), blocks_per_stage: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        stages: List[nn.Module] = []
        ni = widths[0]
        for i, nf in enumerate(widths):
            for b in range(blocks_per_stage):
                stages.append(ResBlock1d(ni, nf, stride=2 if (b == 0 and i > 0) else 1))
                ni = nf
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(ni, num_classes))
        self.feature_dim = ni

    def features(self, x):
        if x.dim() == 4:
            x = x[:, -1]
        return self.stages(self.stem(x)).mean(dim=-1)

    def representation(self, x):
        return self.features(x)

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, -1]
        return self.head(self.stages(self.stem(x)))


class InceptionModule1d(nn.Module):
    def __init__(self, ni, nf=32, ks=(39, 19, 9), bottleneck=32):
        super().__init__()
        self.bottleneck = (nn.Conv1d(ni, bottleneck, 1, bias=False)
                           if ni > 1 else nn.Identity())
        nb = bottleneck if ni > 1 else ni
        self.convs = nn.ModuleList([_conv(nb, nf, k) for k in ks])
        self.maxpool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                     nn.Conv1d(ni, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(ks) + 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b = self.bottleneck(x)
        outs = [c(b) for c in self.convs] + [self.maxpool(x)]
        return self.act(self.bn(torch.cat(outs, dim=1)))


class Inception1d(nn.Module):
    """inception1d supervised baseline (PTB-XL benchmark family)."""

    def __init__(self, n_leads: int = 8, num_classes: int = 5, depth: int = 6, nf: int = 32):
        super().__init__()
        layers, ni = [], n_leads
        for _ in range(depth):
            layers.append(InceptionModule1d(ni, nf=nf))
            ni = nf * 4
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(ni, num_classes))
        self.feature_dim = ni

    def representation(self, x):
        if x.dim() == 4:
            x = x[:, -1]
        return self.body(x).mean(dim=-1)

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, -1]
        return self.head(self.body(x))
