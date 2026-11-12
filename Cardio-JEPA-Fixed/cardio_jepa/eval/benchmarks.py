"""Downstream benchmarks from Section 6.2.

  1. ``run_classification_probe``  -- PTB-XL / CPSC2018 multi-label (sanity check)
  2. ``run_anticipation_probe``    -- near-term rhythm/arrhythmia onset (H1)
  3. ``run_burden_probe``          -- AF-burden regression (H2)

Every probe runs on a **frozen** encoder. For Cardio-JEPA the feature vector can
optionally be extended with the predictor's forecast latents
(``feature_mode="world"``), which is the operational form of the world-model
claim: the anticipation head sees not only the current cardiac state but where
the model thinks that state is heading.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import EvalConfig, SignalConfig
from ..data.longform import AnticipationDataset, LongRecord, WindowSample
from ..data.preprocess import expand_to_n_leads, prepare_window
from .linear_probe import macro_auroc, probe_encoder, train_linear_probe
from .stats import bootstrap_auroc


def make_feature_fn(model, horizons_s: Sequence[float] = (),
                    mode: str = "context") -> Callable[[torch.Tensor], torch.Tensor]:
    """Frozen-encoder feature extractor.

    mode='context': mean-pooled encoder embedding only (works for every model,
                    including the ST-MEM and supervised baselines).
    mode='world':   [context embedding | predicted future latent per horizon],
                    available only for models with a forward predictor.
    """
    if mode not in ("context", "world"):
        raise ValueError(f"Unknown feature mode: {mode}")
    if mode == "world" and (not hasattr(model, "anticipation_features") or not horizons_s):
        raise ValueError("world features require a predictor and at least one horizon")
    if mode == "world" and hasattr(model, "anticipation_features") and horizons_s:
        def fn(x: torch.Tensor) -> torch.Tensor:
            return model.anticipation_features(x, horizons_s)
        return fn

    def fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            b, w, c, t = x.shape
            return model.representation(x.reshape(b*w, c, t)).reshape(b, w, -1).mean(1)
        return model.representation(x)
    return fn


@torch.no_grad()
def _extract(feature_fn, ds, device, batch_size: int = 64, num_workers: int = 0
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, labels = [], []
    for x, y in loader:
        f = feature_fn(x.to(device))
        feats.append(f.detach().float().cpu())
        labels.append(y.detach().cpu())
    if not feats:
        return torch.zeros(0), torch.zeros(0)
    return torch.cat(feats), torch.cat(labels)


# --------------------------------------------------------------------------------------
# 1. standard diagnostic classification
# --------------------------------------------------------------------------------------
def run_classification_probe(model, splits: Dict[str, object], device,
                             eval_cfg: EvalConfig, batch_size: int = 64,
                             num_workers: int = 0, verbose: bool = True
                             ) -> Dict[str, object]:
    model.to(device).eval()
    fn = make_feature_fn(model, mode="context")
    res = probe_encoder(fn, splits["train"], splits["test"],
                        n_classes=int(splits["n_classes"]), device=device,
                        batch_size=batch_size, epochs=eval_cfg.linear_probe_epochs,
                        lr=eval_cfg.linear_probe_lr, multilabel=True,
                        label_fractions=eval_cfg.label_fractions,
                        num_workers=num_workers, verbose=verbose)
    res["label_names"] = splits.get("label_names")
    return res


# --------------------------------------------------------------------------------------
# 2. anticipation
# --------------------------------------------------------------------------------------
def run_anticipation_probe(model, records: Sequence[LongRecord],
                           train_samples: Sequence[WindowSample],
                           test_samples: Sequence[WindowSample],
                           signal: SignalConfig, device, eval_cfg: EvalConfig,
                           context_windows: int = 1,
                           feature_mode: str = "world",
                           horizons_s: Sequence[float] = (),
                           batch_size: int = 64, num_workers: int = 0,
                           verbose: bool = True) -> Dict[str, object]:
    """Binary anticipation probe on a frozen encoder. Returns AUROC + bootstrap CI."""
    model.to(device).eval()
    if not train_samples or not test_samples:
        return {"auroc": float("nan"), "note": "insufficient samples"}

    tr_ds = AnticipationDataset(records, train_samples, signal, context_windows)
    te_ds = AnticipationDataset(records, test_samples, signal, context_windows)
    fn = make_feature_fn(model, horizons_s, mode=feature_mode)

    Ftr, Ytr = _extract(fn, tr_ds, device, batch_size, num_workers)
    Fte, Yte = _extract(fn, te_ds, device, batch_size, num_workers)
    if len(Ftr) == 0 or len(Fte) == 0 or (len(np.unique(Yte.numpy())) < 2 or len(np.unique(Ytr.numpy())) < 2):
        return {"auroc": float("nan"), "note": "degenerate split (single-class test set)"}

    by_frac: Dict[float, float] = {}
    best = None
    for frac in eval_cfg.label_fractions:
        r = train_linear_probe(Ftr, Ytr.unsqueeze(-1), Fte, Yte.unsqueeze(-1),
                               n_classes=1, epochs=eval_cfg.linear_probe_epochs,
                               lr=eval_cfg.linear_probe_lr, device=device,
                               multilabel=True, label_fraction=float(frac),
                               verbose=False)
        by_frac[float(frac)] = r["macro_auroc"]
        if abs(frac - 1.0) < 1e-9:
            best = r
    best = best or r

    y_true = Yte.numpy().reshape(-1)
    y_score = np.asarray(best["y_score"]).reshape(-1)
    ci = bootstrap_auroc(y_true, y_score, n=eval_cfg.bootstrap_n)

    out = {"auroc": ci["auroc"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
           "by_fraction": by_frac, "y_true": y_true, "y_score": y_score,
           "n_train": len(train_samples), "n_test": len(test_samples),
           "prevalence": float(y_true.mean()), "feature_mode": feature_mode,
           "feature_dim": int(Ftr.shape[1])}
    if verbose:
        print(f"    anticipation [{feature_mode}]: AUROC = {ci['auroc']:.4f} "
              f"[{ci['lo']:.4f}, {ci['hi']:.4f}]  "
              f"n_test={len(test_samples)}  prev={out['prevalence']:.3f}")
    return out


# --------------------------------------------------------------------------------------
# 3. AF-burden regression
# --------------------------------------------------------------------------------------
def run_burden_probe(model, records: Sequence[LongRecord],
                     train_samples: Sequence[WindowSample],
                     test_samples: Sequence[WindowSample],
                     signal: SignalConfig, device, context_windows: int = 6,
                     feature_mode: str = "world", horizons_s: Sequence[float] = (),
                     batch_size: int = 32, num_workers: int = 0,
                     alpha: float = 1.0, verbose: bool = True) -> Dict[str, object]:
    """Ridge regression from a frozen embedding to whole-record AF burden."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score

    model.to(device).eval()
    if not train_samples or not test_samples:
        return {"r2": float("nan"), "note": "insufficient samples"}

    tr_ds = AnticipationDataset(records, train_samples, signal, context_windows)
    te_ds = AnticipationDataset(records, test_samples, signal, context_windows)
    fn = make_feature_fn(model, horizons_s, mode=feature_mode)

    Ftr, Ytr = _extract(fn, tr_ds, device, batch_size, num_workers)
    Fte, Yte = _extract(fn, te_ds, device, batch_size, num_workers)
    if len(Ftr) == 0 or len(Fte) == 0:
        return {"r2": float("nan"), "note": "no features"}

    reg = Ridge(alpha=alpha).fit(Ftr.numpy(), Ytr.numpy())
    pred = np.clip(reg.predict(Fte.numpy()), 0.0, 1.0)
    y = Yte.numpy()

    out: Dict[str, object] = {
        "r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "y_true": y, "y_pred": pred, "feature_mode": feature_mode,
        "n_train": len(train_samples), "n_test": len(test_samples)}
    try:
        from scipy.stats import spearmanr
        out["spearman"] = float(spearmanr(y, pred).correlation)
    except Exception:
        out["spearman"] = float("nan")
    if verbose:
        print(f"    AF burden [{feature_mode}]: R2 = {out['r2']:.3f}  "
              f"MAE = {out['mae']:.3f}  rho = {out['spearman']:.3f}")
    return out


# --------------------------------------------------------------------------------------
# forecast skill: is the predictor doing anything beyond copying the present?
# --------------------------------------------------------------------------------------
@torch.no_grad()
def forecast_skill(model, records: Sequence[LongRecord], signal: SignalConfig, device,
                   horizons_s: Sequence[float] = (10.0, 60.0, 300.0, 900.0),
                   n_samples: int = 128, context_windows: int = 1, seed: int = 0,
                   verbose: bool = True):
    """Compare the predictor against a persistence baseline, per horizon.

    The forward objective is only meaningful if `g_phi` beats the trivial solution of
    echoing the present. For each (context, true future) pair we measure

      * ``cos_pred``        cos( g_phi(z_ctx, h),  sg(fbar(x_future)) )
      * ``cos_persistence`` cos( pooled z_ctx,     sg(fbar(x_future)) )

    ``skill = cos_pred - cos_persistence``. Skill <= 0 means the predictor has learned
    nothing a copy of the context would not give you, and any downstream "world model"
    claim is unsupported no matter how good the AUROC looks.

    Both quantities are computed on centred embeddings: raw pooled latents share a large
    constant component that puts every cosine near 1.0 and hides the comparison.
    """
    import pandas as pd

    model.to(device).eval()
    if not hasattr(model, "forward_predictor"):
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    usable = [r for r in records if r.duration_s > 0]
    if not usable:
        return pd.DataFrame()

    rows = []
    for H in horizons_s:
        ctxs, futs = [], []
        need_s = context_windows * signal.window_s + H + signal.window_s
        for _ in range(n_samples):
            cand = [r for r in usable if r.duration_s > need_s]
            if not cand:
                break
            rec = cand[int(rng.integers(len(cand)))]
            t0 = float(rng.uniform(0.0, rec.duration_s - need_s))
            start = int(t0 * rec.fs)
            win_n = int(round(signal.window_s * rec.fs))

            def prep(s):
                raw = rec.read(s, win_n)
                x = prepare_window(raw, rec.fs, signal.fs, signal.window_samples,
                                   reduced=(rec.n_channels >= 12))
                return expand_to_n_leads(x, signal.n_leads)

            ctxs.append(np.stack([prep(start + w * win_n) for w in range(context_windows)]))
            futs.append(prep(start + context_windows * win_n + int(round(H * rec.fs))))

        if len(ctxs) < 8:
            if verbose:
                print(f"  horizon {H:.0f}s: no record long enough, skipping")
            continue

        predictions, truths, copies = [], [], []
        enc = model.target_encoder if getattr(model, "target_encoder", None) is not None else model.encoder
        for offset in range(0, len(ctxs), 8):
            ctx = torch.from_numpy(np.stack(ctxs[offset:offset+8])).float().to(device)
            fut = torch.from_numpy(np.stack(futs[offset:offset+8])).float().to(device)
            predictions.append(model.forecast_latent(ctx, H).float().cpu())
            truths.append(model._target_tokens(fut).mean(1).float().cpu())
            z = model.encoder.full_tokens(ctx.reshape(-1, ctx.size(-2), ctx.size(-1))).mean(1)
            copies.append(z.reshape(len(ctx), context_windows, -1).mean(1).float().cpu())
        pred, true, persist = map(torch.cat, (predictions, truths, copies))

        def centred_cos(a, b):
            a = (a - a.mean(0, keepdim=True)).float()
            b = (b - b.mean(0, keepdim=True)).float()
            return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())

        cp, cq = centred_cos(pred, true), centred_cos(persist, true)
        rows.append({"horizon_s": H, "cos_pred": cp, "cos_persistence": cq,
                     "skill": cp - cq, "n": len(ctxs)})
        if verbose:
            print(f"  horizon {H:6.0f}s  cos_pred={cp:+.4f}  "
                  f"cos_persistence={cq:+.4f}  skill={cp - cq:+.4f}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# horizon sweep (H2)
# --------------------------------------------------------------------------------------
def horizon_sweep(model_fns: Dict[str, object], records: Sequence[LongRecord],
                  signal: SignalConfig, device, eval_cfg: EvalConfig,
                  horizons_s: Sequence[float], builder, context_windows: int = 1,
                  test_frac: float = 0.3, seed: int = 0, verbose: bool = True):
    """Run an anticipation benchmark at several horizons for several models.

    H2 predicts the hierarchical-vs-single-scale gap *widens monotonically* with
    the horizon, so the sweep is the actual test, not a nice-to-have.
    """
    import pandas as pd
    from ..data.longform import split_by_record

    rows = []
    for H in horizons_s:
        samples, stats = builder(records, signal, horizon_s=H,
                                 context_windows=context_windows)
        if stats["n_positive"] < 5 or stats["n_negative"] < 5:
            if verbose:
                print(f"  horizon {H:.0f}s: too few samples "
                      f"(pos={stats['n_positive']:.0f}, neg={stats['n_negative']:.0f}), skipping")
            continue
        tr, te = split_by_record(samples, len(records), test_frac=test_frac, seed=seed,
                                 held_out_records=getattr(eval_cfg, "held_out_records", None))
        for name, spec in model_fns.items():
            model = spec["model"] if isinstance(spec, dict) else spec
            mode = spec.get("feature_mode", "context") if isinstance(spec, dict) else "context"
            r = run_anticipation_probe(model, records, tr, te, signal, device, eval_cfg,
                                       context_windows=context_windows,
                                       feature_mode=mode, horizons_s=[H],
                                       verbose=False)
            rows.append({"horizon_s": H, "model": name, "auroc": r.get("auroc"),
                         "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"),
                         "prevalence": r.get("prevalence"), "n_test": r.get("n_test")})
            if verbose:
                print(f"  horizon {H:6.0f}s  {name:28s} AUROC={r.get('auroc', float('nan')):.4f}")
    return pd.DataFrame(rows)
