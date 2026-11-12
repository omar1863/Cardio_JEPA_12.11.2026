"""Representation-collapse diagnostics (RQ3, proposal Section 5.5).

Logged every epoch across all four anti-collapse arms:

  * **effective rank** via singular-value entropy (RankMe, Garrido et al. 2023):
        p_i = sigma_i / sum_j sigma_j ,   erank = exp(-sum_i p_i log p_i)
    A collapsed representation concentrates all spectral mass on one direction and
    drops to erank ~ 1; a healthy one sits at a decent fraction of ``d``.
  * **per-dimension variance** (mean / min / fraction of near-dead dimensions).
  * **loss-curve shape** -- ER-JEPA (arXiv:2607.01145) reports a specific collapse
    signature where the loss falls to ~0 and later partially recovers. We detect it
    explicitly rather than eyeballing the curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


@torch.no_grad()
def effective_rank(z: torch.Tensor, eps: float = 1e-12, center: bool = False) -> float:
    """Singular-value-entropy effective rank (RankMe) of an (N, D) embedding matrix.

    ``center=False`` (the default, and what RankMe specifies) is what detects
    *complete* collapse: if every sample maps to the same vector, the matrix is
    rank 1 and this returns ~1. Mean-centring first would subtract exactly that
    constant direction away and report the residual noise as full rank, hiding the
    failure we are trying to measure.

    ``center=True`` gives the covariance spectrum instead, which is the right lens
    for *dimensional* collapse (some directions dying while others survive). Both
    are logged; see ``embedding_report``.
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.size(-1))
    z = z.detach().float()
    if z.size(0) < 2:
        return float("nan")
    if center:
        z = z - z.mean(dim=0, keepdim=True)
    try:
        sv = torch.linalg.svdvals(z)
    except Exception:                                  # pragma: no cover - rare LAPACK fail
        return float("nan")
    total = sv.sum()
    if float(total) <= eps:
        return 1.0
    p = sv / total
    entropy = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(entropy))


@torch.no_grad()
def variance_stats(z: torch.Tensor, dead_threshold: float = 1e-3) -> Dict[str, float]:
    if z.dim() == 3:
        z = z.reshape(-1, z.size(-1))
    z = z.detach().float()
    if z.size(0) < 2:
        return {"var_mean": float("nan"), "var_min": float("nan"),
                "std_mean": float("nan"), "dead_dim_frac": float("nan")}
    var = z.var(dim=0, unbiased=False)
    std = var.sqrt()
    return {
        "var_mean": float(var.mean()),
        "var_min": float(var.min()),
        "std_mean": float(std.mean()),
        "dead_dim_frac": float((std < dead_threshold).float().mean()),
    }


@torch.no_grad()
def pairwise_cosine(z: torch.Tensor, max_n: int = 512) -> float:
    """Mean off-diagonal cosine similarity. -> 1.0 means every sample maps to the
    same direction (the textbook complete-collapse signature)."""
    if z.dim() == 3:
        z = z.mean(dim=1)
    z = z.detach().float()[:max_n]
    if z.size(0) < 2:
        return float("nan")
    zn = torch.nn.functional.normalize(z, dim=-1)
    sim = zn @ zn.T
    n = sim.size(0)
    off = (sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))
    return float(off)


@torch.no_grad()
def embedding_report(z: torch.Tensor) -> Dict[str, float]:
    """All RQ3 diagnostics for one batch of embeddings.

    The ratios are normalised by ``min(n_samples, d)``, not by ``d``: the rank of an
    (N, D) matrix cannot exceed ``min(N, D)``, so dividing by ``d`` alone makes every
    arm look collapsed whenever the probe batch is smaller than the embedding width.
    Use a probe batch of at least ``embed_dim`` samples for a well-conditioned estimate.
    """
    flat = z.reshape(-1, z.size(-1)) if z.dim() == 3 else z
    n, d = flat.shape
    denom = max(min(n, d), 1)

    out = {
        "effective_rank": effective_rank(z, center=False),      # complete collapse
        "cov_effective_rank": effective_rank(z, center=True),   # dimensional collapse
        "cos_sim": pairwise_cosine(z),
        "rank_denominator": float(denom),
    }
    out.update(variance_stats(z))
    out["erank_ratio"] = out["effective_rank"] / denom
    out["cov_erank_ratio"] = out["cov_effective_rank"] / denom
    return out


def detect_collapse_signature(losses: Sequence[float],
                              near_zero_frac: float = 0.05,
                              recovery_frac: float = 0.5) -> Dict[str, object]:
    """Detect ER-JEPA's reported 'loss -> ~0 then partial recovery' signature.

    Args:
        losses: per-epoch training loss.
        near_zero_frac: a loss below this fraction of the initial loss counts as
            "collapsed to ~0".
        recovery_frac: a subsequent rise above this fraction of the initial loss
            counts as "recovered".
    """
    losses = [float(x) for x in losses]
    out: Dict[str, object] = {"collapsed_to_zero": False, "recovered": False,
                              "collapse_epoch": None, "recovery_epoch": None}
    if len(losses) < 3:
        return out
    l0 = max(abs(losses[0]), 1e-8)
    for i, l in enumerate(losses):
        if l < near_zero_frac * l0:
            out["collapsed_to_zero"] = True
            out["collapse_epoch"] = i
            for j in range(i + 1, len(losses)):
                if losses[j] > recovery_frac * l0:
                    out["recovered"] = True
                    out["recovery_epoch"] = j
                    break
            break
    return out


@dataclass
class CollapseMonitor:
    """Accumulates per-epoch diagnostics for one training arm."""

    arm: str = "ema"
    history: List[Dict[str, float]] = field(default_factory=list)

    def log(self, epoch: int, loss: float, z: Optional[torch.Tensor] = None,
            extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        rec: Dict[str, float] = {"epoch": float(epoch), "loss": float(loss)}
        if z is not None:
            rec.update(embedding_report(z))
        if extra:
            rec.update({k: float(v) for k, v in extra.items()})
        self.history.append(rec)
        return rec

    @property
    def losses(self) -> List[float]:
        return [r["loss"] for r in self.history]

    def summary(self) -> Dict[str, object]:
        if not self.history:
            return {"arm": self.arm}
        last = self.history[-1]
        first = self.history[0]
        out: Dict[str, object] = {
            "arm": self.arm,
            "final_loss": last.get("loss"),
            "final_effective_rank": last.get("effective_rank"),
            "final_erank_ratio": last.get("erank_ratio"),
            "final_cov_effective_rank": last.get("cov_effective_rank"),
            "final_cov_erank_ratio": last.get("cov_erank_ratio"),
            "final_std_mean": last.get("std_mean"),
            "final_cos_sim": last.get("cos_sim"),
            # Complete collapse = the *centred* spectrum is degenerate too, i.e. there
            # is essentially nothing left for a downstream head. A near-1 cosine
            # similarity on its own is NOT sufficient evidence: mean-pooling 400
            # LayerNormed tokens always leaves a large shared component, and a
            # representation can sit at cos ~ 0.9999 and still linearly separate
            # classes from the residual. That case is reported as `offset_dominated`.
            "complete_collapse": bool(last.get("cov_erank_ratio", 1.0) < 0.02
                                      and last.get("std_mean", 1.0) < 1e-5),
            "dimensional_collapse": bool(last.get("cov_erank_ratio", 1.0) < 0.05),
            "offset_dominated": bool(last.get("cos_sim", 0.0) > 0.99),
            "erank_delta": (last.get("effective_rank", float("nan"))
                            - first.get("effective_rank", float("nan"))),
        }
        out.update(detect_collapse_signature(self.losses))
        # a shrinking effective rank *and* a rising cosine similarity is the
        # dimensional-collapse pattern H3 predicts for the EMA-only arm
        eranks = [r.get("effective_rank", float("nan")) for r in self.history]
        eranks = [e for e in eranks if not np.isnan(e)]
        out["erank_monotone_decreasing"] = bool(
            len(eranks) >= 3 and all(b <= a + 1e-6 for a, b in zip(eranks, eranks[1:])))
        return out

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.history)
