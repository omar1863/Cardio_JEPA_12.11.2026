"""Signal preprocessing shared by every dataset loader."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..config import REDUCED_LEAD_IDX

try:
    from scipy.signal import butter, resample_poly, sosfiltfilt
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample along the last axis. x: (..., T)."""
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32, copy=False)
    if _HAVE_SCIPY:
        from fractions import Fraction
        frac = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
        return resample_poly(x, frac.numerator, frac.denominator,
                             axis=-1).astype(np.float32)
    # linear-interpolation fallback
    n_out = int(round(x.shape[-1] * fs_out / fs_in))
    src = np.linspace(0.0, 1.0, x.shape[-1], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1])
    out = np.stack([np.interp(dst, src, row) for row in flat])
    return out.reshape(*x.shape[:-1], n_out).astype(np.float32)


def bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 45.0,
             order: int = 3) -> np.ndarray:
    """Zero-phase bandpass. Removes baseline wander (<0.5 Hz) and EMG/mains (>45 Hz)."""
    if not _HAVE_SCIPY:
        return x.astype(np.float32, copy=False)
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    if low >= high or x.shape[-1] < 3 * order * 4:
        return x.astype(np.float32, copy=False)
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def reduce_leads(x: np.ndarray, lead_idx: Sequence[int] = REDUCED_LEAD_IDX) -> np.ndarray:
    """12-lead -> 8 independent leads (I, II, V1..V6). x: (..., C, T)."""
    if x.shape[-2] == len(lead_idx):
        return x
    if x.shape[-2] < 12:
        raise ValueError(f"expected 12 leads to reduce, got {x.shape[-2]}")
    return np.take(x, list(lead_idx), axis=-2)


def normalize(x: np.ndarray, mode: str = "instance", eps: float = 1e-6) -> np.ndarray:
    """Per-lead standardisation. x: (..., C, T)."""
    if mode == "none":
        return x
    if mode == "instance":                          # per lead, per recording
        mu = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True)
        return ((x - mu) / (sd + eps)).astype(np.float32)
    if mode == "global":                            # shared across leads
        mu = x.mean(axis=(-2, -1), keepdims=True)
        sd = x.std(axis=(-2, -1), keepdims=True)
        return ((x - mu) / (sd + eps)).astype(np.float32)
    raise ValueError(f"unknown normalize mode {mode!r}")


def clean(x: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """Replace NaN/Inf and clip extreme values left by lead-off artefacts."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -clip, clip).astype(np.float32)


def is_valid(x: np.ndarray, min_std: float = 1e-4) -> bool:
    """Reject flat-line / all-zero / NaN recordings (ECG-JEPA `remove_invalid_samples`)."""
    if not np.isfinite(x).all():
        return False
    if np.abs(x[..., :15]).sum() == 0:
        return False
    return float(x.std()) > min_std


def prepare_window(x: np.ndarray, fs_in: float, fs_out: float,
                   n_samples: int, reduced: bool = True,
                   do_bandpass: bool = True, norm: str = "instance") -> np.ndarray:
    """Full pipeline for one window: (C, T_in) -> (C_out, n_samples) float32."""
    x = np.asarray(x, dtype=np.float32)
    if reduced and x.shape[0] >= 12:
        x = reduce_leads(x)
    x = clean(x)
    if do_bandpass:
        x = bandpass(x, fs_in)
    x = resample_signal(x, fs_in, fs_out)
    if x.shape[-1] < n_samples:
        x = np.pad(x, ((0, 0), (0, n_samples - x.shape[-1])), mode="edge")
    x = x[..., :n_samples]
    return normalize(x, norm)


def expand_to_n_leads(x: np.ndarray, n_leads: int) -> np.ndarray:
    """Tile/truncate the lead axis so single- or two-channel data (MIT-BIH, LTAFDB)
    can be fed to an 8-lead model.

    Long-term Holter databases record 2 channels, not 12. Rather than train a
    separate model we replicate the available channels across the lead axis; the
    encoder's positional embedding still distinguishes the slots, and the lead
    dimension degenerates gracefully. Documented explicitly because it is a real
    modelling compromise, not a no-op.
    """
    c = x.shape[-2]
    if c == n_leads:
        return x
    if c > n_leads:
        return x[..., :n_leads, :]
    reps = int(np.ceil(n_leads / c))
    return np.concatenate([x] * reps, axis=-2)[..., :n_leads, :]
