"""High-level wiring: discovered mounts -> ready-to-train loaders and probe splits.

This is the layer the notebook calls. It degrades gracefully: whatever subset of
the six corpora is attached gets used, and if none is attached (or a corpus fails
to parse) the synthetic generator keeps every downstream cell runnable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..engine import InterleavedLoader
from .discovery import FoundDataset, discover
from .longform import (ForwardPairDataset, LongRecord, collate_mixed, load_long_records,
                       synthetic_long_records)
from .short import (ArrayWindowDataset, ShortIndex, ShortWindowDataset, build_index,
                    ptbxl_splits, wfdb_splits)
from .synthetic import synth_short_corpus


def load_short_indices(found: Dict[str, FoundDataset], cfg: ExperimentConfig,
                       verbose: bool = True) -> Dict[str, ShortIndex]:
    out: Dict[str, ShortIndex] = {}
    seen_roots: Dict[str, str] = {}

    keys = list(cfg.data.pretrain_datasets)
    if cfg.data.use_noise_corpora:
        keys += [k for k, d in found.items() if d.kind == "noise"]

    for key in keys:
        ds = found.get(key)
        if ds is None or ds.kind not in ("short", "noise"):
            continue
        # A combined mirror (e.g. shaoxing-and-ningbo) satisfies several corpus keys
        # from one directory. Indexing it once per key would load every record twice.
        if ds.data_root in seen_roots:
            if verbose:
                print(f"[data] {ds.label:22s} shares a directory with "
                      f"{seen_roots[ds.data_root]} - already indexed, skipping")
            continue
        seen_roots[ds.data_root] = ds.label
        try:
            ix = build_index(ds, cfg.data.max_records_per_dataset)
        except Exception as exc:
            if verbose:
                print(f"[data] {ds.label}: index failed ({type(exc).__name__}: {exc})")
            continue
        if len(ix) == 0:
            if verbose:
                print(f"[data] {ds.label}: no usable records")
            continue
        out[key] = ix
        if verbose:
            print(f"[data] {ds.label:22s} indexed {len(ix):>7d} recordings")
    return out


def load_long_corpus(found: Dict[str, FoundDataset], cfg: ExperimentConfig,
                     verbose: bool = True) -> List[LongRecord]:
    """LTAFDB + MIT-BIH records; falls back to synthetic Holter-style records."""
    records: List[LongRecord] = []
    for key in cfg.data.long_datasets:
        ds = found.get(key)
        if ds is None:
            continue
        try:
            recs = load_long_records(ds, limit=cfg.data.max_long_records)
        except Exception as exc:
            if verbose:
                print(f"[data] {ds.label}: load failed ({type(exc).__name__}: {exc})")
            continue
        with_rhythm = sum(1 for r in recs if r.episodes)
        if verbose:
            print(f"[data] {ds.label:22s} {len(recs):>4d} long records "
                  f"({with_rhythm} with rhythm annotations, "
                  f"total {sum(r.duration_s for r in recs) / 3600:.1f} h)")
        records.extend(recs)

    if not records and cfg.data.allow_synthetic_fallback:
        if verbose:
            print("[data] no long recordings found -> using synthetic Holter records "
                  "(pipeline smoke test only, not a scientific result)")
        records = synthetic_long_records(
            cfg.data.synthetic_n_long, cfg.data.synthetic_long_minutes,
            cfg.signal.fs, cfg.signal.n_leads, seed=cfg.optim.seed)
    return records


def build_pretrain_loaders(found: Dict[str, FoundDataset], cfg: ExperimentConfig,
                           long_records: Optional[Sequence[LongRecord]] = None,
                           samples_per_record: int = 200,
                           steps_per_epoch: Optional[int] = None,
                           verbose: bool = True) -> Tuple[InterleavedLoader, Dict[str, object]]:
    """Interleaved loader over short clips (L_mask + intra-window L_fwd) and long
    recordings (cross-window L_fwd, plus the Level-2 stack when enabled)."""
    info: Dict[str, object] = {}
    loaders: List[DataLoader] = []
    weights: List[float] = []

    # ---- short corpora -----------------------------------------------------------
    indices = load_short_indices(found, cfg, verbose=verbose)
    if indices:
        ds_short = ShortWindowDataset(list(indices.values()), cfg.signal, augment=True)
        info["n_short"] = len(ds_short)
        info["short_sources"] = {k: len(v) for k, v in indices.items()}
    elif cfg.data.allow_synthetic_fallback:
        if verbose:
            print("[data] no short corpora found -> synthetic 10 s clips")
        waves, _ = synth_short_corpus(cfg.data.synthetic_n_short, cfg.signal.fs,
                                      cfg.signal.window_s, cfg.signal.n_leads,
                                      seed=cfg.optim.seed)
        ds_short = ArrayWindowDataset(waves)
        info["n_short"] = len(ds_short)
        info["short_sources"] = {"synthetic": len(ds_short)}
    else:
        ds_short = None

    if ds_short is not None and len(ds_short) > 0:
        loaders.append(DataLoader(ds_short, batch_size=cfg.optim.batch_size, shuffle=True,
                                  num_workers=cfg.optim.num_workers, drop_last=False,
                                  collate_fn=collate_mixed,
                                  pin_memory=torch.cuda.is_available()))
        weights.append(1.0)

    # ---- long corpora ------------------------------------------------------------
    if long_records is None:
        long_records = load_long_corpus(found, cfg, verbose=verbose)
    if long_records:
        ds_long = ForwardPairDataset(
            long_records, cfg.signal,
            context_windows=cfg.forward.context_windows,
            horizons_s=cfg.forward.horizons_s,
            samples_per_record=samples_per_record,
            l2_horizons_s=(cfg.hierarchical.l2_horizons_s
                           if cfg.hierarchical.enabled else None),
            seed=cfg.optim.seed)
        info["n_forward_pairs"] = len(ds_long)
        if len(ds_long) > 0:
            loaders.append(DataLoader(ds_long, batch_size=cfg.optim.batch_size,
                                      shuffle=True, num_workers=cfg.optim.num_workers,
                                      drop_last=False, collate_fn=collate_mixed,
                                      pin_memory=torch.cuda.is_available()))
            # the forward objective is the contribution under test: weight it up
            weights.append(2.0)
        elif verbose:
            print("[data] long records are too short for the configured horizons; "
                  "L_fwd will use the intra-window path only")

    if not loaders:
        raise RuntimeError(
            "No training data available. Attach at least one ECG dataset under "
            "/kaggle/input, or set cfg.data.allow_synthetic_fallback = True.")

    loader = InterleavedLoader(loaders, weights, steps_per_epoch=steps_per_epoch)
    info["steps_per_epoch"] = len(loader)
    info["long_records"] = long_records
    if verbose:
        print(f"[data] {len(loaders)} loader(s), {len(loader)} steps/epoch, "
              f"batch={cfg.optim.batch_size}")
    return loader, info


def build_probe_splits(found: Dict[str, FoundDataset], cfg: ExperimentConfig,
                       verbose: bool = True) -> Dict[str, Dict[str, object]]:
    """Supervised splits for the standard-classification sanity check."""
    out: Dict[str, Dict[str, object]] = {}

    if "ptbxl" in found:
        try:
            ix = build_index(found["ptbxl"], cfg.data.max_records_per_dataset)
            out["ptbxl"] = ptbxl_splits(ix, cfg.signal)
            if verbose:
                print(f"[probe] PTB-XL: {len(out['ptbxl']['train'])} train / "
                      f"{len(out['ptbxl']['test'])} test, "
                      f"{out['ptbxl']['n_classes']} superdiagnostic classes")
        except Exception as exc:
            if verbose:
                print(f"[probe] PTB-XL split failed: {type(exc).__name__}: {exc}")

    if "cpsc" in found:
        try:
            ix = build_index(found["cpsc"], cfg.data.max_records_per_dataset)
            out["cpsc"] = wfdb_splits(ix, cfg.signal)
            if verbose:
                print(f"[probe] CPSC2018: {len(out['cpsc']['train'])} train / "
                      f"{len(out['cpsc']['test'])} test, "
                      f"{out['cpsc']['n_classes']} classes")
        except Exception as exc:
            if verbose:
                print(f"[probe] CPSC2018 split failed: {type(exc).__name__}: {exc}")

    if not out and cfg.data.allow_synthetic_fallback:
        if verbose:
            print("[probe] no labelled corpus found -> synthetic AF/non-AF task")
        n = max(256, cfg.data.synthetic_n_short)
        w, y = synth_short_corpus(n, cfg.signal.fs, cfg.signal.window_s,
                                  cfg.signal.n_leads, seed=cfg.optim.seed + 7)
        # Single column, not a 2-column one-hot: AF vs sinus is one binary label, and
        # encoding it as two perfectly anti-correlated columns makes the macro-AUROC
        # average a score with its own near-complement, which is harder to read and
        # buys nothing.
        Y = y.reshape(-1, 1).astype(np.int64)
        cut = int(len(w) * 0.7)
        out["synthetic"] = {
            "train": ArrayWindowDataset(w[:cut], Y[:cut], return_dict=False),
            "test": ArrayWindowDataset(w[cut:], Y[cut:], return_dict=False),
            "val": None, "label_names": ["AF"], "n_classes": 1}
    return out


def probe_batch_from_loader(loader, n: int = 128) -> torch.Tensor:
    """A fixed batch of windows for the per-epoch collapse diagnostics."""
    chunks: List[torch.Tensor] = []
    total = 0
    for batch in loader:
        x = batch["ctx"] if isinstance(batch, dict) else batch[0]
        x = x[:, -1] if x.dim() == 4 else x
        chunks.append(x)
        total += len(x)
        if total >= n:
            break
    if not chunks:
        raise RuntimeError("could not draw a probe batch: loader is empty")
    return torch.cat(chunks)[:n]
