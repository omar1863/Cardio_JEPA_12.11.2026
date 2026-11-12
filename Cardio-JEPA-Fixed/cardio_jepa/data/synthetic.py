"""Synthetic ECG generator.

Purpose: make the notebook runnable end-to-end with **zero** attached datasets, so
every code path (pretraining, all four RQ3 arms, the hierarchical stack, the
anticipation benchmarks) can be smoke-tested before a GPU quota is spent.

The generator produces a P-QRS-T complex as a sum of Gaussians (the standard
McSharry-style parameterisation), with two rhythm states:

  * **sinus**  -- regular RR with respiratory sinus arrhythmia, P wave present
  * **AF**     -- irregularly irregular RR, P wave replaced by fibrillatory waves

Long synthetic records alternate between the two so the AF-onset anticipation
benchmark has something to detect. These signals are *not* a substitute for real
data: they exist to validate the pipeline, and any number computed on them should
be read as a smoke-test result, not as evidence about ECG.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# (amplitude, centre in seconds relative to R, width in seconds)
_WAVES_SINUS = [
    (0.10, -0.20, 0.025),    # P
    (-0.10, -0.035, 0.008),  # Q
    (1.20, 0.000, 0.010),    # R
    (-0.25, 0.035, 0.012),   # S
    (0.30, 0.200, 0.045),    # T
]
_WAVES_AF = [
    (-0.10, -0.035, 0.008),
    (1.15, 0.000, 0.010),
    (-0.25, 0.035, 0.012),
    (0.28, 0.200, 0.045),
]

# per-lead amplitude/polarity scaling for the 8 reduced leads (I, II, V1..V6)
_LEAD_GAIN = np.array([0.7, 1.0, -0.4, -0.2, 0.5, 0.9, 1.0, 0.8], dtype=np.float32)


def _beat(t: np.ndarray, waves) -> np.ndarray:
    out = np.zeros_like(t)
    for amp, mu, sigma in waves:
        out += amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    return out


def _rr_intervals(duration_s: float, rhythm: str, hr: float,
                  rng: np.random.Generator) -> np.ndarray:
    base = 60.0 / hr
    rr: List[float] = []
    total = 0.0
    while total < duration_s:
        if rhythm == "af":
            # irregularly irregular: heavy-tailed multiplicative jitter
            interval = base * float(np.clip(rng.lognormal(0.0, 0.22), 0.45, 2.2))
        else:
            # sinus with respiratory modulation (~0.25 Hz) + small noise
            phase = 2 * np.pi * 0.25 * total
            interval = base * (1.0 + 0.05 * np.sin(phase)) * float(
                np.clip(rng.normal(1.0, 0.025), 0.8, 1.25))
        rr.append(interval)
        total += interval
    return np.asarray(rr, dtype=np.float64)


def synth_segment(duration_s: float, fs: float, rhythm: str = "sinus",
                  hr: float = 70.0, n_leads: int = 8,
                  rng: Optional[np.random.Generator] = None,
                  noise: float = 0.03) -> np.ndarray:
    """Return (n_leads, duration_s*fs) synthetic ECG."""
    rng = rng or np.random.default_rng()
    n = int(round(duration_s * fs))
    trace = np.zeros(n, dtype=np.float32)

    waves = _WAVES_AF if rhythm == "af" else _WAVES_SINUS
    half = 0.35                                   # seconds of beat template each side
    tt = np.arange(-half, half, 1.0 / fs)
    template = _beat(tt, waves).astype(np.float32)

    pos = 0.0
    for interval in _rr_intervals(duration_s, rhythm, hr, rng):
        centre = int(round(pos * fs))
        lo = centre - len(template) // 2
        a, b = max(0, lo), min(n, lo + len(template))
        if b > a:
            trace[a:b] += template[a - lo:b - lo]
        pos += interval

    if rhythm == "af":
        # fibrillatory baseline: narrowband 5-9 Hz wander replacing organised P waves
        t = np.arange(n) / fs
        f = rng.uniform(5.0, 9.0)
        trace += (0.05 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))).astype(np.float32)

    sig = np.outer(_LEAD_GAIN[:n_leads] if n_leads <= 8
                   else np.resize(_LEAD_GAIN, n_leads), trace).astype(np.float32)

    # baseline wander + white noise, per lead
    t = np.arange(n) / fs
    for c in range(sig.shape[0]):
        sig[c] += 0.08 * np.sin(2 * np.pi * rng.uniform(0.15, 0.4) * t
                                + rng.uniform(0, 6.28)).astype(np.float32)
        sig[c] += rng.normal(0.0, noise, size=n).astype(np.float32)
    return sig


def synth_short_corpus(n: int, fs: float = 250.0, window_s: float = 10.0,
                       n_leads: int = 8, af_frac: float = 0.25,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """n synthetic 10 s clips. Returns (waves (n, C, T), labels (n,) 1 = AF)."""
    rng = np.random.default_rng(seed)
    waves, labels = [], []
    for _ in range(n):
        is_af = rng.random() < af_frac
        hr = rng.uniform(90, 150) if is_af else rng.uniform(52, 95)
        waves.append(synth_segment(window_s, fs, "af" if is_af else "sinus", hr,
                                   n_leads, rng))
        labels.append(int(is_af))
    return np.stack(waves), np.asarray(labels, dtype=np.int64)


def synth_long_record(minutes: float = 40.0, fs: float = 250.0, n_leads: int = 8,
                      seed: int = 0, af_burden: Optional[float] = None,
                      af_episode_s: Tuple[float, float] = (30.0, 180.0)
                      ) -> Dict[str, object]:
    """One long synthetic Holter-style record alternating sinus <-> AF.

    ``af_burden`` is the target fraction of the record spent in AF; if None it is
    drawn per record so the corpus spans everything from AF-free to AF-dominated.
    Sinus episode lengths are then set so the realised burden matches, which is what
    makes *paroxysmal* AF (rare, short bursts inside long sinus stretches) possible.

    This matters for the anticipation benchmark: with equal-length alternating
    episodes, every sinus window has an AF onset close behind it and the negative
    class is empty, so the benchmark is undefined. Long sinus stretches are what
    create genuine negatives.

    Returns dict with ``signal`` (C, T), ``fs`` and ``episodes``: a list of
    ``(start_sample, end_sample, rhythm)`` triples in the same shape the LTAFDB
    loader produces, so downstream code is identical for real and synthetic data.
    """
    rng = np.random.default_rng(seed)
    total_s = minutes * 60.0

    if af_burden is None:
        # Paroxysmal regime: low burden, so sinus stretches are long relative to the
        # forecast horizons. A record only contains *both* anticipation classes when
        # its sinus stretches exceed the horizon -- some windows then sit far from any
        # onset (negatives) and some sit just before one (positives). Sampling burden
        # uniformly over [0, 0.9] instead yields records that are all-positive or
        # all-negative, and a record-level split of those cannot produce a usable
        # test set. 10% of records are AF-free throughout.
        af_burden = 0.0 if rng.random() < 0.10 else float(0.03 + 0.25 * rng.beta(1.5, 2.5))
    af_burden = float(np.clip(af_burden, 0.0, 0.9))

    a_bar = float(np.mean(af_episode_s))
    # E[sinus] chosen so E[af] / (E[af] + E[sinus]) == af_burden
    s_bar = total_s * 2 if af_burden <= 1e-6 else a_bar * (1.0 - af_burden) / af_burden

    segments, episodes = [], []
    pos_s, cursor = 0.0, 0
    rhythm = "sinus"                       # always start in sinus

    while pos_s < total_s:
        if rhythm == "af":
            dur = float(rng.uniform(*af_episode_s))
        else:
            dur = float(rng.uniform(0.5 * s_bar, 1.5 * s_bar))
        dur = min(dur, total_s - pos_s)
        if dur <= 1.0:
            break

        hr = rng.uniform(95, 155) if rhythm == "af" else rng.uniform(50, 90)
        seg = synth_segment(dur, fs, rhythm, hr, n_leads, rng)
        segments.append(seg)
        episodes.append((cursor, cursor + seg.shape[1], "AFIB" if rhythm == "af" else "N"))
        cursor += seg.shape[1]
        pos_s += dur
        rhythm = "af" if rhythm == "sinus" else "sinus"

    signal = (np.concatenate(segments, axis=1) if segments
              else np.zeros((n_leads, int(total_s * fs)), dtype=np.float32))
    return {"signal": signal, "fs": fs, "episodes": episodes,
            "name": f"synth{seed:03d}", "n_samples": signal.shape[1],
            "target_af_burden": af_burden}


def synth_long_corpus(n_records: int, minutes: float = 40.0, fs: float = 250.0,
                      n_leads: int = 8, seed: int = 0) -> List[Dict[str, object]]:
    return [synth_long_record(minutes, fs, n_leads, seed=seed + i) for i in range(n_records)]
