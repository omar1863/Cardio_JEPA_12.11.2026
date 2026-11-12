"""Anti-collapse mechanisms compared under RQ3 (proposal Section 5.5).

Four arms:
  1. ``ema``    -- encoder asymmetry + stop-gradient only (Kim / Weimann & Conrad).
  2. ``vicreg`` -- variance hinge + covariance penalty (BRo-JEPA style).
  3. ``vq``     -- soft vector-quantised codebook bottleneck on the target
                   embeddings (MTS-JEPA style).
  4. ``sigreg`` -- LeJEPA / Sketched Isotropic Gaussian Regularization: push the
                   embedding distribution towards an isotropic Gaussian using
                   sliced univariate normality tests, which removes the need for a
                   second (target) network altogether.

Every arm exposes the same interface::

    reg = build_regularizer(loss_cfg, embed_dim)
    penalty, stats = reg(embeddings)          # embeddings: (B, D) or (B, N, D)
    target = reg.transform_target(target)     # identity except for the VQ arm
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten(z: torch.Tensor) -> torch.Tensor:
    """(B, D) or (B, N, D) -> (B*N, D) in float32 (penalties are scale sensitive)."""
    if z.dim() == 3:
        z = z.reshape(-1, z.size(-1))
    return z.float()


class BaseRegularizer(nn.Module):
    name = "none"

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        return z.new_zeros(()), {}

    def transform_target(self, h: torch.Tensor) -> torch.Tensor:
        return h


class EMAOnly(BaseRegularizer):
    """Arm 1 -- no explicit penalty; collapse avoidance relies on EMA + stop-grad."""
    name = "ema"


class VICReg(BaseRegularizer):
    """Arm 2 -- variance/covariance regularisation.

        L_reg = w_v * mean_k max(0, gamma - std(z_k))
              + w_c * (1/d) * sum_{i != j} Cov(z)_ij^2
    """

    name = "vicreg"

    def __init__(self, gamma: float = 1.0, var_weight: float = 1.0,
                 cov_weight: float = 0.04):
        super().__init__()
        self.gamma = gamma
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, z: torch.Tensor):
        z = _flatten(z)
        n, d = z.shape
        if n < 2:
            return z.new_zeros(()), {}
        z = z - z.mean(dim=0, keepdim=True)

        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(self.gamma - std).mean()

        cov = (z.T @ z) / (n - 1)
        off_diag = cov - torch.diag_embed(torch.diagonal(cov))
        cov_loss = off_diag.pow(2).sum() / d

        loss = self.var_weight * var_loss + self.cov_weight * cov_loss
        return loss, {"vicreg_var": float(var_loss.detach()),
                      "vicreg_cov": float(cov_loss.detach()),
                      "vicreg_std_mean": float(std.mean().detach())}


class SoftVQ(BaseRegularizer):
    """Arm 3 -- differentiable soft codebook bottleneck on the *target* embeddings.

    Targets are replaced by a temperature-softmaxed convex combination of codebook
    entries before the prediction loss is taken. A constant target then requires a
    constant code assignment, which the batch-level entropy bonus penalises, so the
    trivial solution stops being free.
    """

    name = "vq"

    def __init__(self, embed_dim: int, codebook_size: int = 512,
                 temperature: float = 0.5, entropy_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.codebook_size = codebook_size
        self.codebook = nn.Parameter(F.normalize(torch.randn(codebook_size, embed_dim),
                                                 dim=-1))
        self._last_assign = None

    def _assign(self, h: torch.Tensor) -> torch.Tensor:
        # Cosine logits rather than raw euclidean distance: with d=384 embeddings the
        # squared distances are O(d), so a temperature of 0.5 on them saturates the
        # softmax and kills the gradient that is supposed to spread codebook usage.
        flat = F.normalize(h.reshape(-1, h.size(-1)).float(), dim=-1)
        book = F.normalize(self.codebook.float(), dim=-1)
        logits = (flat @ book.T) / max(self.temperature, 1e-4)
        return F.softmax(logits, dim=-1)

    def transform_target(self, h: torch.Tensor) -> torch.Tensor:
        assign = self._assign(h)
        self._last_assign = assign          # kept live: the entropy term trains the codebook
        quant = assign @ self.codebook.to(assign.dtype)
        # keep the target's original scale so the prediction loss stays comparable
        # across arms (the codebook only reshapes *where* targets can live)
        quant = quant * (h.reshape(-1, h.size(-1)).float().norm(dim=-1, keepdim=True)
                         / (quant.norm(dim=-1, keepdim=True) + 1e-6))
        # Detach the *target*: if the prediction loss could reach the codebook, the
        # cheapest way to cut it is to move the targets towards whatever the predictor
        # already outputs -- a target-chases-prediction shortcut that is collapse by
        # another name. The codebook is shaped only by the spread/commitment penalty
        # in `forward`, which still sees a live `_last_assign`.
        return quant.reshape(h.shape).to(h.dtype).detach()

    def forward(self, z: torch.Tensor):
        if self._last_assign is None:
            return z.new_zeros(()), {}
        assign = self._last_assign
        log_k = math.log(self.codebook_size)

        # (a) usage spread: penalise *low* marginal entropy. The previous
        # `ent_sample - ent_batch` form is exactly 0 both at full collapse (every
        # sample on one code: 0 - 0) and cannot punish it, which is the failure this
        # arm exists to prevent. `log K - ent_batch` is >= 0 and maximal at collapse.
        marginal = assign.mean(dim=0)
        ent_batch = -(marginal * torch.log(marginal + 1e-8)).sum()
        spread = (log_k - ent_batch) / log_k

        # (b) commitment: each individual target should still pick a few codes
        ent_sample = -(assign * torch.log(assign + 1e-8)).sum(dim=-1).mean()
        commit = ent_sample / log_k

        loss = self.entropy_weight * (spread + 0.1 * commit)
        perplexity = float(torch.exp(ent_batch).detach())
        self._last_assign = None
        return loss, {"vq_perplexity": perplexity,
                      "vq_spread": float(spread.detach()),
                      "vq_ent_sample": float(ent_sample.detach())}


class SIGReg(BaseRegularizer):
    """Arm 4 -- Sketched Isotropic Gaussian Regularization (LeJEPA).

    Sketch the embedding distribution onto ``n_slices`` random unit directions and
    penalise the deviation of each 1-D projection from a standard normal using an
    Epps--Pulley style characteristic-function statistic:

        T(u) = sum_t w_t * | (1/n) sum_i exp(i t u_i) - exp(-t^2/2) |^2

    Zero only when every projection is N(0, 1); by Cramer--Wold that pins the joint
    distribution to an isotropic Gaussian, which is full rank by construction. This
    is the arm that lets us drop the EMA target network entirely.
    """

    name = "sigreg"

    def __init__(self, embed_dim: int, n_slices: int = 64, n_freqs: int = 16):
        super().__init__()
        self.n_slices = n_slices
        self.embed_dim = embed_dim
        freqs = torch.linspace(0.25, 3.0, n_freqs)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, z: torch.Tensor):
        z = _flatten(z)
        n, d = z.shape
        if n < 4:
            return z.new_zeros(()), {}

        # sketch: fresh random directions every step (this is the "sketched" part)
        dirs = torch.randn(d, self.n_slices, device=z.device, dtype=z.dtype)
        dirs = F.normalize(dirs, dim=0)
        proj = z @ dirs                                     # (n, n_slices)

        # standardise each projection so the test targets shape, not scale/offset,
        # then separately penalise the scale so variance cannot shrink to zero
        mu = proj.mean(dim=0, keepdim=True)
        sd = proj.std(dim=0, keepdim=True) + 1e-5
        u = (proj - mu) / sd

        t = self.freqs.to(z.dtype).view(1, 1, -1)           # (1, 1, F)
        arg = u.unsqueeze(-1) * t                           # (n, S, F)
        emp_re = torch.cos(arg).mean(dim=0)                 # (S, F)
        emp_im = torch.sin(arg).mean(dim=0)
        target_re = torch.exp(-0.5 * self.freqs.to(z.dtype) ** 2).view(1, -1)
        stat = ((emp_re - target_re) ** 2 + emp_im ** 2).mean()

        # scale term: log-variance should sit at 0 (unit variance)
        scale_pen = (torch.log(sd.squeeze(0) ** 2 + 1e-8) ** 2).mean()

        loss = stat + 0.1 * scale_pen
        return loss, {"sigreg_stat": float(stat.detach()),
                      "sigreg_scale": float(scale_pen.detach())}


def build_regularizer(loss_cfg, embed_dim: int) -> BaseRegularizer:
    arm = loss_cfg.anti_collapse
    if arm == "ema":
        return EMAOnly()
    if arm == "vicreg":
        return VICReg(gamma=loss_cfg.vicreg_var_gamma,
                      var_weight=loss_cfg.vicreg_var_weight,
                      cov_weight=loss_cfg.vicreg_cov_weight)
    if arm == "vq":
        return SoftVQ(embed_dim=embed_dim,
                      codebook_size=loss_cfg.vq_codebook_size,
                      temperature=loss_cfg.vq_temperature,
                      entropy_weight=loss_cfg.vq_entropy_weight)
    if arm == "sigreg":
        return SIGReg(embed_dim=embed_dim,
                      n_slices=loss_cfg.sigreg_n_slices,
                      n_freqs=loss_cfg.sigreg_n_freqs)
    raise ValueError(f"unknown anti_collapse arm: {arm!r} "
                     f"(expected one of: ema, vicreg, vq, sigreg)")
