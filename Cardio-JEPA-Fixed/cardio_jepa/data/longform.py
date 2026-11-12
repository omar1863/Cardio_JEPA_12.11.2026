"""Long-recording (Holter / patch) data: LTAFDB and MIT-BIH.

These are the corpora the proposal marks **Required** for the forward-prediction
and hierarchical objectives (Table 6.1), because they are the only ones long
enough to contain a genuine "future window".

Provides
  * ``LongRecord``          -- uniform random-access view over a long recording,
                               with rhythm episodes and beat annotations.
  * ``ForwardPairDataset``  -- (context strip, future window, horizon) pretraining pairs.
  * ``build_onset_dataset`` -- the near-term rhythm-onset anticipation benchmark
                               (proposal Section 6.2 items 2-3) with explicit
                               leakage control.
  * ``build_burden_dataset``-- AF-burden regression (Section 6.2 item 4, for H2).

A note on leads: LTAFDB and MIT-BIH are 2-channel. ``expand_to_n_leads`` tiles
those channels across the 8-lead axis so a single encoder serves both regimes;
this is a documented modelling compromise, not a claim that the tiled channels
carry 12-lead information.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import SignalConfig
from . import wfdb_lite
from .discovery import FoundDataset
from .preprocess import expand_to_n_leads, is_valid, prepare_window

AF_RHYTHMS = frozenset({"AFIB", "AFL"})
# WFDB aux rhythm strings look like "(AFIB"; normalise by stripping the paren.
def _norm_rhythm(aux: str) -> str:
    return aux.strip().lstrip("(").strip().upper()


@dataclass
class LongRecord:
    """Random-access view over one long recording."""

    name: str
    fs: float
    n_samples: int
    n_channels: int
    path: Optional[str] = None                 # WFDB record path (no extension)
    memory: Optional[np.ndarray] = None        # (C, T) for synthetic/in-memory records
    episodes: List[Tuple[int, int, str]] = field(default_factory=list)
    beat_samples: Optional[np.ndarray] = None
    beat_symbols: Optional[List[str]] = None

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def read(self, start: int, n: int) -> np.ndarray:
        """Native-rate raw window, shape (C, n). Out-of-range reads are edge-padded."""
        start = max(0, min(start, max(0, self.n_samples - 1)))
        stop = min(self.n_samples, start + n)
        if self.memory is not None:
            sig = np.asarray(self.memory[:, start:stop], dtype=np.float32)
        else:
            rec = wfdb_lite.read_record(self.path, sampfrom=start, sampto=stop)
            sig = np.asarray(rec.p_signal, dtype=np.float32).T
        if sig.shape[1] < n and sig.shape[1] > 0:
            sig = np.pad(sig, ((0, 0), (0, n - sig.shape[1])), mode="edge")
        elif sig.shape[1] == 0:
            sig = np.zeros((self.n_channels, n), dtype=np.float32)
        return sig

    # ---------------------------------------------------------------- rhythm views
    def rhythm_intervals(self, rhythms: Sequence[str] = tuple(AF_RHYTHMS)
                         ) -> List[Tuple[int, int]]:
        want = {r.upper() for r in rhythms}
        return [(s, e) for s, e, r in self.episodes if r.upper() in want]

    def rhythm_at(self, sample: int) -> Optional[str]:
        for s, e, r in self.episodes:
            if s <= sample < e:
                return r
        return None

    def burden(self, rhythms: Sequence[str] = tuple(AF_RHYTHMS)) -> float:
        """Fraction of the record spent in the given rhythms (AF burden)."""
        if self.n_samples <= 0:
            return 0.0
        total = sum(e - s for s, e in self.rhythm_intervals(rhythms))
        return float(np.clip(total / self.n_samples, 0.0, 1.0))

    def beats_in(self, start: int, stop: int) -> List[str]:
        if self.beat_samples is None or self.beat_symbols is None:
            return []
        lo = int(np.searchsorted(self.beat_samples, start, "left"))
        hi = int(np.searchsorted(self.beat_samples, stop, "left"))
        return self.beat_symbols[lo:hi]


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def _episodes_from_annotation(ann: wfdb_lite.Annotation, n_samples: int
                              ) -> List[Tuple[int, int, str]]:
    """Turn '+' rhythm-change annotations into contiguous (start, end, rhythm) spans."""
    marks: List[Tuple[int, str]] = []
    for smp, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        if sym == "+" and aux:
            marks.append((int(smp), _norm_rhythm(aux)))
    if not marks:
        return []
    out: List[Tuple[int, int, str]] = []
    for i, (smp, rhythm) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n_samples
        if end > smp and rhythm:
            out.append((smp, end, rhythm))
    return out


def load_long_records(ds: FoundDataset, limit: int = 40,
                      want_beats: bool = True) -> List[LongRecord]:
    """Load LTAFDB / MIT-BIH records (annotations only -- signals stay on disk)."""
    records: List[LongRecord] = []
    if ds.fmt == "mitbih_csv":
        return _load_mitbih_csv(ds, limit=limit)

    paths = wfdb_lite.list_records(ds.data_root, ("hea",), limit=None)
    for p in paths:
        if len(records) >= limit:
            break
        try:
            hdr = wfdb_lite.parse_header(p + ".hea")
        except Exception:
            continue
        if hdr.n_samp <= 0 or hdr.fs <= 0:
            continue

        episodes: List[Tuple[int, int, str]] = []
        beats_s, beats_y = None, None
        for ext in ("atr", "qrs"):
            if not os.path.exists(p + "." + ext):
                continue
            try:
                ann = wfdb_lite.read_annotation(p, ext)
            except Exception:
                continue
            if ext == "atr":
                episodes = _episodes_from_annotation(ann, hdr.n_samp)
                if want_beats:
                    keep = [i for i, s in enumerate(ann.symbol)
                            if s in wfdb_lite.BEAT_SYMBOLS]
                    if keep:
                        beats_s = ann.sample[keep]
                        beats_y = [ann.symbol[i] for i in keep]
            break

        records.append(LongRecord(name=os.path.basename(p), fs=float(hdr.fs),
                                  n_samples=int(hdr.n_samp), n_channels=hdr.n_sig,
                                  path=p, episodes=episodes,
                                  beat_samples=beats_s, beat_symbols=beats_y))
    return records


def _load_mitbih_csv(ds: FoundDataset, limit: int = 40) -> List[LongRecord]:
    """The ``protobioengineering/mit-bih-...-modern-2023`` CSV mirror.

    Layout: ``{id}_ekg.csv`` (2 signal columns + a ``symbol`` column) and
    ``{id}_annotations.csv``. Sampling rate is 360 Hz.
    """
    import pandas as pd

    ekgs: Dict[str, str] = {}
    anns: Dict[str, List[str]] = {}
    for dirpath, _d, filenames in os.walk(ds.root):
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            if fn.endswith("_ekg.csv"):
                ekgs[fn[: -len("_ekg.csv")]] = full
            elif "_annotations" in fn and fn.endswith(".csv"):
                anns.setdefault(fn.split("_annotations")[0], []).append(full)

    records: List[LongRecord] = []
    for rid in sorted(ekgs)[:limit]:
        try:
            df = pd.read_csv(ekgs[rid])
        except Exception:
            continue
        num = df.select_dtypes(include=[np.number])
        drop = [c for c in num.columns
                if c.lower() in ("sample", "sample #", "time", "index", "unnamed: 0")]
        num = num.drop(columns=drop, errors="ignore")
        if num.shape[1] < 1:
            continue
        sig = num.to_numpy(dtype=np.float32).T                 # (C, T)

        episodes: List[Tuple[int, int, str]] = []
        beats_s, beats_y = None, None
        if rid in anns:
            try:
                adf = pd.read_csv(sorted(anns[rid])[0])
            except Exception:
                adf = None
            if adf is not None:
                scol = next((c for c in adf.columns
                             if c.lower() in ("sample", "sample #", "index")), None)
                ycol = next((c for c in adf.columns
                             if c.lower() in ("symbol", "type", "label")), None)
                acol = next((c for c in adf.columns
                             if c.lower() in ("aux", "aux_note", "auxnote")), None)
                if scol and ycol:
                    smp = adf[scol].to_numpy(dtype=np.int64)
                    sym = adf[ycol].astype(str).tolist()
                    aux = (adf[acol].fillna("").astype(str).tolist() if acol
                           else [""] * len(sym))
                    ann = wfdb_lite.Annotation(sample=smp, symbol=sym, aux_note=aux,
                                               fs=360.0)
                    episodes = _episodes_from_annotation(ann, sig.shape[1])
                    keep = [i for i, s in enumerate(sym) if s in wfdb_lite.BEAT_SYMBOLS]
                    if keep:
                        beats_s = smp[keep]
                        beats_y = [sym[i] for i in keep]

        records.append(LongRecord(name=rid, fs=360.0, n_samples=sig.shape[1],
                                  n_channels=sig.shape[0], memory=sig,
                                  episodes=episodes, beat_samples=beats_s,
                                  beat_symbols=beats_y))
    return records


def synthetic_long_records(n: int, minutes: float, fs: float, n_leads: int,
                           seed: int = 0) -> List[LongRecord]:
    from .synthetic import synth_long_corpus
    out: List[LongRecord] = []
    for d in synth_long_corpus(n, minutes=minutes, fs=fs, n_leads=n_leads, seed=seed):
        sig = d["signal"]
        out.append(LongRecord(name=str(d["name"]), fs=float(d["fs"]),
                              n_samples=int(sig.shape[1]), n_channels=int(sig.shape[0]),
                              memory=sig, episodes=list(d["episodes"])))
    return out


# --------------------------------------------------------------------------------------
# pretraining pairs
# --------------------------------------------------------------------------------------
class ForwardPairDataset(Dataset):
    """(context strip, future window, horizon) samples for L_fwd.

    A sample is ``W`` consecutive windows ending at time t, plus the window that
    starts ``horizon`` seconds after t. When the hierarchical stack is enabled an
    extra far-future window is drawn at the Level-2 horizon.
    """

    def __init__(self, records: Sequence[LongRecord], signal: SignalConfig,
                 context_windows: int = 1,
                 horizons_s: Sequence[float] = (10.0, 30.0, 60.0, 300.0),
                 samples_per_record: int = 200,
                 l2_horizons_s: Optional[Sequence[float]] = None,
                 seed: int = 0):
        self.records = [r for r in records if r.duration_s > 0]
        self.signal = signal
        self.W = max(1, context_windows)
        self.horizons = list(horizons_s)
        self.l2_horizons = list(l2_horizons_s) if l2_horizons_s else None
        self.rng = np.random.default_rng(seed)

        self.plan: List[Tuple[int, int, float, Optional[float]]] = []
        win_s = signal.window_s
        for ri, rec in enumerate(self.records):
            ctx_s = self.W * win_s
            for _ in range(samples_per_record):
                h = float(self.rng.choice(self.horizons))
                l2h = (float(self.rng.choice(self.l2_horizons))
                       if self.l2_horizons else None)
                need_s = ctx_s + max(h, l2h or 0.0) + win_s
                if rec.duration_s <= need_s:
                    # record too short for this horizon: fall back to the largest
                    # horizon it can support, or skip it entirely
                    usable = rec.duration_s - ctx_s - win_s
                    feasible = [x for x in self.horizons if x < usable]
                    if not feasible:
                        continue
                    h = float(max(feasible))
                    l2h = None
                    need_s = ctx_s + h + win_s
                start_s = float(self.rng.uniform(0.0, rec.duration_s - need_s))
                self.plan.append((ri, int(start_s * rec.fs), h, l2h))

    def __len__(self) -> int:
        return len(self.plan)

    def _prep(self, rec: LongRecord, start: int) -> np.ndarray:
        n_native = int(round(self.signal.window_s * rec.fs))
        raw = rec.read(start, n_native)
        x = prepare_window(raw, rec.fs, self.signal.fs, self.signal.window_samples,
                           reduced=(rec.n_channels >= 12))
        x = expand_to_n_leads(x, self.signal.n_leads)
        return x if is_valid(x) else np.zeros_like(x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ri, start, h, l2h = self.plan[idx]
        rec = self.records[ri]
        win_native = int(round(self.signal.window_s * rec.fs))

        ctx = np.stack([self._prep(rec, start + w * win_native) for w in range(self.W)])
        ctx_end = start + self.W * win_native
        fut = self._prep(rec, ctx_end + int(round(h * rec.fs)))

        out = {
            "ctx": torch.from_numpy(ctx).float(),                 # (W, C, T)
            "future": torch.from_numpy(fut).float(),              # (C, T)
            "has_future": torch.tensor(1, dtype=torch.bool),
            "horizon_s": torch.tensor(float(h)),
        }
        if l2h is not None:
            l2f = self._prep(rec, ctx_end + int(round(l2h * rec.fs)))
            out["l2_future"] = torch.from_numpy(l2f).float()
            out["l2_horizon_s"] = torch.tensor(float(l2h))
            out["has_l2"] = torch.tensor(1, dtype=torch.bool)
        return out


def collate_mixed(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate short (no future) and long (with future) items into one batch.

    Items without a future window contribute only to L_mask and the intra-window
    forward term; the loop below splits the batch by ``has_future`` so the two
    regimes never get mixed inside a single loss term.
    """
    out: Dict[str, torch.Tensor] = {}
    widths = {int(b["ctx"].shape[0]) for b in batch}
    W = max(widths)
    ctx = []
    for b in batch:
        c = b["ctx"]
        if c.shape[0] < W:                      # repeat the earliest window as padding
            c = torch.cat([c[:1].repeat(W - c.shape[0], 1, 1), c], dim=0)
        ctx.append(c)
    out["ctx"] = torch.stack(ctx)
    out["has_future"] = torch.stack([b["has_future"] for b in batch])
    out["horizon_s"] = torch.stack([b["horizon_s"] for b in batch])
    if any("future" in b for b in batch):
        zero = torch.zeros_like(batch[0]["ctx"][0])
        out["future"] = torch.stack([b.get("future", zero) for b in batch])
    if any("l2_future" in b for b in batch):
        zero = torch.zeros_like(batch[0]["ctx"][0])
        out["l2_future"] = torch.stack([b.get("l2_future", zero) for b in batch])
        out["l2_horizon_s"] = torch.stack(
            [b.get("l2_horizon_s", torch.tensor(0.0)) for b in batch])
        out["has_l2"] = torch.stack(
            [b.get("has_l2", torch.tensor(0, dtype=torch.bool)) for b in batch])
    if any("label" in b for b in batch):
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# --------------------------------------------------------------------------------------
# downstream benchmarks
# --------------------------------------------------------------------------------------
@dataclass
class WindowSample:
    record: int
    start: int                 # sample index of the context start (native rate)
    label: float


class AnticipationDataset(Dataset):
    """Frozen-encoder probe inputs for the anticipation benchmarks."""

    def __init__(self, records: Sequence[LongRecord], samples: Sequence[WindowSample],
                 signal: SignalConfig, context_windows: int = 1):
        self.records = list(records)
        self.samples = list(samples)
        self.signal = signal
        self.W = max(1, context_windows)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rec = self.records[s.record]
        win_native = int(round(self.signal.window_s * rec.fs))
        wins = []
        for w in range(self.W):
            raw = rec.read(s.start + w * win_native, win_native)
            x = prepare_window(raw, rec.fs, self.signal.fs, self.signal.window_samples,
                               reduced=(rec.n_channels >= 12))
            wins.append(expand_to_n_leads(x, self.signal.n_leads))
        ctx = np.stack(wins)
        return torch.from_numpy(ctx).float(), torch.tensor(s.label, dtype=torch.float32)


def build_onset_dataset(records: Sequence[LongRecord], signal: SignalConfig,
                        horizon_s: float, context_windows: int = 1,
                        stride_s: float = 30.0, guard_s: float = 5.0,
                        rhythms: Sequence[str] = tuple(AF_RHYTHMS),
                        max_per_record: int = 400,
                        negatives_before_first_event_only: bool = False,
                        seed: int = 0) -> Tuple[List[WindowSample], Dict[str, float]]:
    """Near-term rhythm-onset anticipation (proposal Section 6.2, items 2-3).

    For every candidate context window that is itself entirely free of the target
    rhythm, the label is 1 if an episode of that rhythm *starts* within
    ``horizon_s`` after the window ends.

    Leakage control (proposal Section 9):
      * the context window must not overlap the target rhythm at all;
      * a ``guard_s`` band immediately after the context is excluded, so a window
        whose event begins essentially at its edge is dropped rather than being
        called a negative;
      * a negative requires a full ``horizon_s`` of *observed* signal after the
        window -- windows near the end of a record cannot be negatives, because
        the absence of an event there is unobserved rather than known;
      * ``negatives_before_first_event_only`` reproduces the strict Attia-style
        rule: once a patient has had a documented episode, later windows are no
        longer eligible as negatives.
    """
    rng = np.random.default_rng(seed)
    out: List[WindowSample] = []
    n_pos = n_neg = 0

    ctx_s = context_windows * signal.window_s
    for ri, rec in enumerate(records):
        iv = sorted(rec.rhythm_intervals(rhythms))
        if not rec.episodes:
            continue
        starts = np.array([s for s, _e in iv], dtype=np.int64)
        first_event = int(starts[0]) if len(starts) else None

        step = max(1, int(round(stride_s * rec.fs)))
        ctx_n = int(round(ctx_s * rec.fs))
        hor_n = int(round(horizon_s * rec.fs))
        guard_n = int(round(guard_s * rec.fs))

        cands = list(range(0, max(0, rec.n_samples - ctx_n - hor_n), step))
        rng.shuffle(cands)
        taken = 0
        for t0 in cands:
            if taken >= max_per_record:
                break
            t1 = t0 + ctx_n
            # (1) context must be free of the target rhythm
            if any(s < t1 and e > t0 for s, e in iv):
                continue
            # (2) guard band
            if len(starts) and np.any((starts >= t1) & (starts < t1 + guard_n)):
                continue
            # (3) event inside the horizon?
            hit = bool(len(starts) and np.any(
                (starts >= t1 + guard_n) & (starts <= t1 + hor_n)))
            if hit:
                out.append(WindowSample(ri, t0, 1.0))
                n_pos += 1
                taken += 1
            else:
                # (4) negatives need the full horizon observed
                if t1 + hor_n > rec.n_samples:
                    continue
                if (negatives_before_first_event_only and first_event is not None
                        and t0 >= first_event):
                    continue
                out.append(WindowSample(ri, t0, 0.0))
                n_neg += 1
                taken += 1

    stats = {"n_positive": float(n_pos), "n_negative": float(n_neg),
             "prevalence": float(n_pos / max(n_pos + n_neg, 1)),
             "horizon_s": float(horizon_s)}
    return out, stats


def build_burden_dataset(records: Sequence[LongRecord], signal: SignalConfig,
                         context_windows: int = 6, from_first_minutes: float = 30.0,
                         rhythms: Sequence[str] = tuple(AF_RHYTHMS),
                         samples_per_record: int = 8,
                         seed: int = 0) -> Tuple[List[WindowSample], Dict[str, float]]:
    """AF-burden regression (Section 6.2 item 4, for H2).

    Target = fraction of the *whole* record spent in AF; input = an early context
    segment drawn from the first ``from_first_minutes`` minutes. Predicting a
    whole-record property from an early window is exactly the long-horizon
    dependency H2 says the hierarchical model should capture.
    """
    rng = np.random.default_rng(seed)
    out: List[WindowSample] = []
    burdens: List[float] = []
    ctx_n_s = context_windows * signal.window_s

    for ri, rec in enumerate(records):
        if not rec.episodes or rec.duration_s < ctx_n_s * 2:
            continue
        b = rec.burden(rhythms)
        burdens.append(b)
        hi = min(from_first_minutes * 60.0, rec.duration_s - ctx_n_s)
        if hi <= 0:
            continue
        for _ in range(samples_per_record):
            t0 = float(rng.uniform(0.0, hi))
            out.append(WindowSample(ri, int(t0 * rec.fs), b))

    stats = {"n_records": float(len(burdens)),
             "mean_burden": float(np.mean(burdens)) if burdens else 0.0,
             "n_samples": float(len(out))}
    return out, stats


def build_arrhythmia_onset_dataset(records: Sequence[LongRecord], signal: SignalConfig,
                                   horizon_s: float, context_windows: int = 1,
                                   stride_s: float = 10.0, guard_s: float = 2.0,
                                   target_beats: Sequence[str] = ("V", "E", "F"),
                                   max_per_record: int = 600,
                                   seed: int = 0
                                   ) -> Tuple[List[WindowSample], Dict[str, float]]:
    """Beat-level near-term arrhythmia anticipation (MIT-BIH).

    Given a context window containing *no* beat of the target class, predict
    whether one occurs within ``horizon_s``. Same guard-band and observed-horizon
    rules as ``build_onset_dataset``.
    """
    rng = np.random.default_rng(seed)
    want = set(target_beats)
    out: List[WindowSample] = []
    n_pos = n_neg = 0

    ctx_s = context_windows * signal.window_s
    for ri, rec in enumerate(records):
        if rec.beat_samples is None or rec.beat_symbols is None:
            continue
        ev = np.asarray([s for s, y in zip(rec.beat_samples, rec.beat_symbols)
                         if y in want], dtype=np.int64)
        if len(ev) == 0:
            continue

        ctx_n = int(round(ctx_s * rec.fs))
        hor_n = int(round(horizon_s * rec.fs))
        guard_n = int(round(guard_s * rec.fs))
        step = max(1, int(round(stride_s * rec.fs)))

        cands = list(range(0, max(0, rec.n_samples - ctx_n - hor_n), step))
        rng.shuffle(cands)
        taken = 0
        for t0 in cands:
            if taken >= max_per_record:
                break
            t1 = t0 + ctx_n
            if np.any((ev >= t0) & (ev < t1)):          # context must be clean
                continue
            if np.any((ev >= t1) & (ev < t1 + guard_n)):
                continue
            hit = bool(np.any((ev >= t1 + guard_n) & (ev <= t1 + hor_n)))
            if hit:
                out.append(WindowSample(ri, t0, 1.0))
                n_pos += 1
            else:
                if t1 + hor_n > rec.n_samples:
                    continue
                out.append(WindowSample(ri, t0, 0.0))
                n_neg += 1
            taken += 1

    stats = {"n_positive": float(n_pos), "n_negative": float(n_neg),
             "prevalence": float(n_pos / max(n_pos + n_neg, 1)),
             "horizon_s": float(horizon_s)}
    return out, stats


def split_by_record(samples: Sequence[WindowSample], n_records: int,
                    test_frac: float = 0.3, seed: int = 0, stratify: bool = True,
                    held_out_records=None
                    ) -> Tuple[List[WindowSample], List[WindowSample]]:
    """Record-disjoint split. Multiple records per patient require patient-ID grouping.

    With ``stratify=True`` records are ordered by their positive-window rate and the
    test records are drawn evenly across that ordering, so both sides span the range
    from event-free to event-heavy patients. A plain random split over a few dozen
    records can easily hand the test set only AF-heavy patients, which makes AUROC
    undefined (single class) or wildly optimistic; that is a property of the split,
    not of the model.
    """
    if held_out_records is not None:
        test_ids = set(held_out_records)
        return ([s for s in samples if s.record not in test_ids],
                [s for s in samples if s.record in test_ids])
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be in (0, 1)")
    ids = sorted({s.record for s in samples})
    if any(i < 0 or i >= n_records for i in ids):
        raise ValueError("sample record index out of range")
    if len(ids) < 2:
        return list(samples), []
    rng = np.random.default_rng(seed)
    n_test = max(1, min(len(ids)-1, round(len(ids)*test_frac)))
    if stratify:
        rates = {i: np.mean([s.label for s in samples if s.record == i]) for i in ids}
        order = sorted(rng.permutation(ids).tolist(), key=lambda i: rates[i])
        picks = np.linspace(0, len(ids)-1, n_test).round().astype(int)
        test_ids = {order[i] for i in picks}
    else:
        test_ids = set(rng.permutation(ids)[:n_test].tolist())
    return ([s for s in samples if s.record not in test_ids],
            [s for s in samples if s.record in test_ids])
