"""Loaders for the 10-second, 12-lead corpora.

Covers every "short" row of the proposal's Table 6.1:
MIMIC-IV-ECG, CODE-15, Chapman-Shaoxing, Ningbo, CPSC2018 and PTB-XL.

Everything is *lazy*: we build an index of record locators and decode signals in
``__getitem__``. MIMIC-IV-ECG alone is ~800k 10-second 12-lead recordings -- roughly
200 GB as float32 -- so eager loading is not an option at any session RAM size.
``DataConfig.max_records_per_dataset`` additionally caps how many locators are indexed.
"""

from __future__ import annotations

import ast
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig, SignalConfig
from . import wfdb_lite
from .discovery import FoundDataset
from .preprocess import expand_to_n_leads, is_valid, prepare_window

# SNOMED-CT -> label name for the nine CPSC2018 classes (used for display; the
# multi-label matrix itself is built generically from whatever Dx codes appear).
SNOMED_NAMES: Dict[str, str] = {
    "426783006": "SNR", "164889003": "AF", "164890007": "AFL",
    "270492004": "IAVB", "164909002": "LBBB", "59118001": "RBBB",
    "713427006": "CRBBB", "713426002": "IRBBB", "284470004": "PAC",
    "63593006": "SVPB", "164884008": "PVC", "427172004": "PVC",
    "429622005": "STD", "164931005": "STE", "39732003": "LAD",
    "164934002": "TAB", "111975006": "LQT", "164917005": "QAb",
    "47665007": "RAD", "251146004": "LQRSV", "446358003": "RAH",
    "365413008": "LVH", "426177001": "SB", "427084000": "STach",
    "426761007": "SVT", "17338001": "VPB",
}


@dataclass
class ShortIndex:
    """Locators for one corpus. ``fmt`` decides how a locator is decoded."""

    key: str
    label: str
    fmt: str                                  # 'wfdb' | 'ptbxl' | 'hdf5' | 'npy'
    locators: List[object] = field(default_factory=list)
    fs_hint: Optional[float] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.locators)


# --------------------------------------------------------------------------------------
# index builders
# --------------------------------------------------------------------------------------
#: Records in the MIT-BIH Noise Stress Test database that are *pure noise*, not ECG:
#: baseline wander, electrode motion and muscle artefact. Feeding these to an ECG
#: encoder would be training on noise labelled as signal.
NOISE_ONLY_RECORDS = frozenset({"bw", "em", "ma"})


def index_wfdb(ds: FoundDataset, limit: int) -> ShortIndex:
    recs = wfdb_lite.list_records(ds.data_root, ("hea",), limit=None,
                                  exclude_dirs=ds.exclude_dirs)
    if ds.kind == "noise":
        recs = [r for r in recs if os.path.basename(r).lower() not in NOISE_ONLY_RECORDS]
    return ShortIndex(key=ds.key, label=ds.label, fmt="wfdb", locators=recs[:limit])


def index_hdf5(ds: FoundDataset, limit: int) -> ShortIndex:
    """CODE-15 ships HDF5 files with a (N, 4096, 12) ``tracings`` array at 400 Hz."""
    import h5py
    files = []
    _excl = {os.path.abspath(p) for p in ds.exclude_dirs}
    for dirpath, dirnames, filenames in os.walk(ds.root):
        dirnames[:] = [d for d in dirnames
                       if os.path.abspath(os.path.join(dirpath, d)) not in _excl]
        for fn in sorted(filenames):
            if fn.lower().endswith((".hdf5", ".h5")):
                files.append(os.path.join(dirpath, fn))
    locators: List[Tuple[str, int]] = []
    for f in sorted(files):
        try:
            with h5py.File(f, "r") as h:
                key = "tracings" if "tracings" in h else next(iter(h.keys()))
                n = h[key].shape[0]
        except Exception:
            continue
        for i in range(n):
            locators.append((f, i))
            if len(locators) >= limit:
                return ShortIndex(key=ds.key, label=ds.label, fmt="hdf5",
                                  locators=locators, fs_hint=400.0)
    return ShortIndex(key=ds.key, label=ds.label, fmt="hdf5", locators=locators,
                      fs_hint=400.0)


def index_npy(ds: FoundDataset, limit: int) -> ShortIndex:
    files = []
    _excl = {os.path.abspath(p) for p in ds.exclude_dirs}
    for dirpath, dirnames, filenames in os.walk(ds.root):
        dirnames[:] = [d for d in dirnames
                       if os.path.abspath(os.path.join(dirpath, d)) not in _excl]
        for fn in sorted(filenames):
            if fn.lower().endswith(".npy"):
                files.append(os.path.join(dirpath, fn))
    locators: List[Tuple[str, int]] = []
    for f in sorted(files):
        try:
            arr = np.load(f, mmap_mode="r")
        except Exception:
            continue
        if arr.ndim == 2:                         # a single recording per file
            locators.append((f, -1))
        elif arr.ndim == 3:                       # (N, C, T) or (N, T, C)
            for i in range(arr.shape[0]):
                locators.append((f, i))
                if len(locators) >= limit:
                    break
        if len(locators) >= limit:
            break
    return ShortIndex(key=ds.key, label=ds.label, fmt="npy", locators=locators[:limit],
                      fs_hint=500.0)


def index_ptbxl(ds: FoundDataset, limit: int, use_hr: bool = False) -> ShortIndex:
    """Index PTB-XL from ``ptbxl_database.csv`` and attach superdiagnostic labels."""
    import pandas as pd

    root = ds.data_root
    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)

    col = "filename_hr" if use_hr else "filename_lr"
    if col not in db.columns:
        col = "filename_lr" if "filename_lr" in db.columns else "filename_hr"
    fs_hint = 500.0 if col == "filename_hr" else 100.0

    agg_path = os.path.join(root, "scp_statements.csv")
    y_cols: List[str] = []
    Y = None
    if os.path.exists(agg_path):
        agg = pd.read_csv(agg_path, index_col=0)
        agg = agg[agg.diagnostic == 1]

        def superclass(codes: dict) -> List[str]:
            out = {agg.loc[k].diagnostic_class for k in codes if k in agg.index}
            return sorted(x for x in out if isinstance(x, str))

        labels = db.scp_codes.apply(superclass)
        y_cols = sorted({c for row in labels for c in row})
        Y = np.zeros((len(db), len(y_cols)), dtype=np.int64)
        idx = {c: i for i, c in enumerate(y_cols)}
        for r, row in enumerate(labels):
            for c in row:
                Y[r, idx[c]] = 1

    keep = np.arange(len(db))
    if limit < len(db):
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(db), size=limit, replace=False))

    locators = [os.path.join(root, p) for p in db[col].values[keep]]
    locators = [p[:-4] if p.endswith(".dat") else p for p in locators]

    meta: Dict[str, object] = {
        "strat_fold": db.strat_fold.values[keep] if "strat_fold" in db.columns else None,
        "patient_id": db.patient_id.values[keep] if "patient_id" in db.columns else None,
        "label_names": y_cols,
        "labels": None if Y is None else Y[keep],
        "ecg_id": db.index.values[keep],
    }
    return ShortIndex(key=ds.key, label=ds.label, fmt="ptbxl", locators=locators,
                      fs_hint=fs_hint, meta=meta)


def build_index(ds: FoundDataset, limit: int) -> ShortIndex:
    if ds.fmt == "ptbxl":
        return index_ptbxl(ds, limit)
    if ds.fmt == "hdf5":
        return index_hdf5(ds, limit)
    if ds.fmt == "npy":
        return index_npy(ds, limit)
    if ds.fmt == "wfdb":
        return index_wfdb(ds, limit)
    raise ValueError(f"cannot build a short-window index for fmt={ds.fmt!r}")


# --------------------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------------------
def _load_locator(fmt: str, loc, fs_hint: Optional[float]) -> Tuple[np.ndarray, float]:
    """Return (C, T) raw signal in physical units and its sampling rate."""
    if fmt in ("wfdb", "ptbxl"):
        rec = wfdb_lite.read_record(str(loc))
        return np.asarray(rec.p_signal, dtype=np.float32).T, float(rec.fs)

    if fmt == "hdf5":
        import h5py
        path, i = loc
        with h5py.File(path, "r") as h:
            key = "tracings" if "tracings" in h else next(iter(h.keys()))
            arr = np.asarray(h[key][i], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T                                  # (T, C) -> (C, T)
        return arr, float(fs_hint or 400.0)

    if fmt == "npy":
        path, i = loc
        arr = np.load(path, mmap_mode="r")
        arr = np.asarray(arr if i < 0 else arr[i], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr, float(fs_hint or 500.0)

    raise ValueError(f"unknown fmt {fmt!r}")


class ShortWindowDataset(Dataset):
    """Lazily decoded 10 s windows, shaped (1, C, T) so they stack with context strips.

    Yields a dict so a single collate function can serve short clips (no future
    window) and long-recording pairs (with one) in the same training loop.
    """

    def __init__(self, indices: Sequence[ShortIndex], signal: SignalConfig,
                 labels: Optional[np.ndarray] = None, augment: bool = False,
                 return_dict: bool = True):
        self.signal = signal
        self.return_dict = return_dict
        self.augment = augment
        self.items: List[Tuple[str, object, Optional[float]]] = []
        self.source: List[str] = []
        for ix in indices:
            for loc in ix.locators:
                self.items.append((ix.fmt, loc, ix.fs_hint))
                self.source.append(ix.key)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.items)

    def _window(self, idx: int) -> np.ndarray:
        fmt, loc, fs_hint = self.items[idx]
        try:
            raw, fs = _load_locator(fmt, loc, fs_hint)
        except Exception:
            return np.zeros((self.signal.n_leads, self.signal.window_samples),
                            dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] < 8:
            return np.zeros((self.signal.n_leads, self.signal.window_samples),
                            dtype=np.float32)

        need = int(round(self.signal.window_s * fs))
        if raw.shape[1] > need:
            start = (random.randint(0, raw.shape[1] - need) if self.augment
                     else (raw.shape[1] - need) // 2)
            raw = raw[:, start:start + need]

        x = prepare_window(raw, fs, self.signal.fs, self.signal.window_samples,
                           reduced=self.signal.reduced_lead)
        x = expand_to_n_leads(x, self.signal.n_leads)
        if not is_valid(x):
            x = np.zeros_like(x)
        return x

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._window(idx)).float()
        if not self.return_dict:
            if self.labels is not None:
                return x, torch.as_tensor(self.labels[idx])
            return x
        out = {
            "ctx": x.unsqueeze(0),                       # (1, C, T)
            "has_future": torch.tensor(0, dtype=torch.bool),
            "horizon_s": torch.tensor(0.0),
        }
        if self.labels is not None:
            out["label"] = torch.as_tensor(self.labels[idx])
        return out


class ArrayWindowDataset(Dataset):
    """In-memory (N, C, T) windows -- used for synthetic data and cached probes."""

    def __init__(self, waves: np.ndarray, labels: Optional[np.ndarray] = None,
                 return_dict: bool = True):
        self.waves = np.asarray(waves, dtype=np.float32)
        self.labels = labels
        self.return_dict = return_dict

    def __len__(self) -> int:
        return len(self.waves)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.waves[idx]).float()
        if not self.return_dict:
            if self.labels is not None:
                return x, torch.as_tensor(self.labels[idx])
            return x
        out = {"ctx": x.unsqueeze(0),
               "has_future": torch.tensor(0, dtype=torch.bool),
               "horizon_s": torch.tensor(0.0)}
        if self.labels is not None:
            out["label"] = torch.as_tensor(self.labels[idx])
        return out


# --------------------------------------------------------------------------------------
# supervised splits for the downstream benchmarks
# --------------------------------------------------------------------------------------
def ptbxl_splits(ix: ShortIndex, signal: SignalConfig, test_fold: int = 10,
                 val_fold: int = 9) -> Dict[str, object]:
    """Official PTB-XL folds: 1-8 train, 9 val, 10 test (Strodthoff et al.)."""
    Y = ix.meta.get("labels")
    folds = ix.meta.get("strat_fold")
    if Y is None or folds is None:
        raise ValueError("PTB-XL index is missing labels or strat_fold")
    folds = np.asarray(folds)
    tr = np.where(folds < val_fold)[0]
    va = np.where(folds == val_fold)[0]
    te = np.where(folds == test_fold)[0]

    def subset(sel):
        sub = ShortIndex(key=ix.key, label=ix.label, fmt=ix.fmt,
                         locators=[ix.locators[i] for i in sel], fs_hint=ix.fs_hint)
        return ShortWindowDataset([sub], signal, labels=Y[sel], return_dict=False)

    return {"train": subset(tr), "val": subset(va), "test": subset(te),
            "label_names": ix.meta.get("label_names"), "n_classes": Y.shape[1]}


def wfdb_dx_labels(ix: ShortIndex, min_count: int = 30
                   ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Multi-label matrix from the ``#Dx:`` header comments (CPSC2018 / Chapman / Ningbo).

    Returns (Y, label_names, keep_idx). Codes rarer than ``min_count`` are dropped,
    then records left with no positive label are removed.
    """
    per_record: List[List[str]] = []
    for loc in ix.locators:
        codes: List[str] = []
        try:
            hdr = wfdb_lite.parse_header(str(loc) + ".hea")
            for c in hdr.comments:
                if c.lower().startswith("dx"):
                    payload = c.split(":", 1)[1] if ":" in c else ""
                    codes = [s.strip() for s in payload.split(",") if s.strip()]
                    break
        except Exception:
            pass
        per_record.append(codes)

    counts: Dict[str, int] = {}
    for codes in per_record:
        for c in codes:
            counts[c] = counts.get(c, 0) + 1
    names = sorted([c for c, n in counts.items() if n >= min_count])
    if not names:
        names = sorted(counts, key=counts.get, reverse=True)[:10]

    idx = {c: i for i, c in enumerate(names)}
    Y = np.zeros((len(per_record), len(names)), dtype=np.int64)
    for r, codes in enumerate(per_record):
        for c in codes:
            if c in idx:
                Y[r, idx[c]] = 1
    keep = np.where(Y.sum(axis=1) > 0)[0]
    pretty = [f"{SNOMED_NAMES.get(c, c)}" for c in names]
    return Y[keep], pretty, keep


def wfdb_splits(ix: ShortIndex, signal: SignalConfig, test_frac: float = 0.2,
                seed: int = 0, min_count: int = 30) -> Dict[str, object]:
    """Random record-level split for CPSC2018 / Chapman / Ningbo multi-label probing."""
    Y, names, keep = wfdb_dx_labels(ix, min_count=min_count)
    locs = [ix.locators[i] for i in keep]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(locs))
    n_test = max(1, int(len(locs) * test_frac))
    te, tr = perm[:n_test], perm[n_test:]

    def subset(sel):
        sub = ShortIndex(key=ix.key, label=ix.label, fmt=ix.fmt,
                         locators=[locs[i] for i in sel], fs_hint=ix.fs_hint)
        return ShortWindowDataset([sub], signal, labels=Y[sel], return_dict=False)

    return {"train": subset(tr), "test": subset(te), "val": None,
            "label_names": names, "n_classes": Y.shape[1]}
