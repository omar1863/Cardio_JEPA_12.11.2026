"""Configuration objects for Cardio-JEPA.

All experiment knobs live here so that the Kaggle notebook can switch between a
fast smoke run (minutes) and a full pretraining run (hours) by swapping a single
object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Signal geometry
# --------------------------------------------------------------------------------------
# Following Kim's ECG-JEPA (arXiv:2410.08559) we work with the 8 independent leads
# (I, II, V1..V6); leads III, aVR, aVL, aVF are exact linear combinations of I and II
# (Einthoven / Goldberger) and therefore carry no extra information.
REDUCED_LEAD_NAMES: Tuple[str, ...] = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
# Index of those leads inside the standard 12-lead ordering
# [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
REDUCED_LEAD_IDX: Tuple[int, ...] = (0, 1, 6, 7, 8, 9, 10, 11)

# Horizon buckets used by the forward predictor's learned horizon token (Section 5.3b of
# the proposal). Values are in seconds.
HORIZON_BUCKETS_S: Tuple[float, ...] = (
    2.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0)


@dataclass
class SignalConfig:
    """Tokenisation / resampling geometry for a single ECG window."""

    fs: int = 250                 # working sampling rate (Hz)
    window_s: float = 10.0        # seconds per window
    patch_samples: int = 50       # samples per patch  (0.2 s @ 250 Hz)
    n_leads: int = 8              # 8 reduced leads
    reduced_lead: bool = True

    @property
    def window_samples(self) -> int:
        return int(round(self.fs * self.window_s))

    @property
    def n_time_patches(self) -> int:
        assert self.window_samples % self.patch_samples == 0, (
            f"window_samples ({self.window_samples}) must be divisible by "
            f"patch_samples ({self.patch_samples})"
        )
        return self.window_samples // self.patch_samples

    @property
    def n_tokens(self) -> int:
        return self.n_leads * self.n_time_patches


@dataclass
class ModelConfig:
    """Encoder / predictor sizes."""

    encoder_embed_dim: int = 384
    encoder_depth: int = 8
    encoder_num_heads: int = 6
    predictor_embed_dim: int = 192
    predictor_depth: int = 4
    predictor_num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    init_std: float = 0.02
    pos_type: str = "sincos"          # 'sincos' | 'learnable'
    use_cropa: bool = True            # Cross-Pattern Attention in the encoder
    # masked-infilling branch
    mask_type: str = "block"          # 'block' | 'random'
    mask_scale: Tuple[float, float] = (0.175, 0.225)
    # forward branch
    forward_target: str = "tokens"    # 'tokens' (per-patch future latents) | 'pooled'
    predictor_cropa: bool = False     # apply CroPA inside the predictor too


@dataclass
class LossConfig:
    """Dual objective weights + anti-collapse arm (RQ3)."""

    lambda_mask: float = 1.0          # weight on L_mask (morphology)
    weight_forward: float = 1.0       # weight on L_fwd (dynamics)
    beta_reg: float = 1.0             # weight on L_reg (anti-collapse)
    loss_fn: str = "smooth_l1"        # 'smooth_l1' | 'l1' | 'l2'
    # RQ3 arm: 'ema' | 'vicreg' | 'vq' | 'sigreg'
    anti_collapse: str = "ema"
    #: What L_reg is measured on: 'pooled' (per-sample representation, what every
    #: downstream probe consumes), 'tokens' (flattened patch latents), or 'both'.
    #: 'tokens' is misleading on its own -- positional variance satisfies the penalty
    #: while every sample still maps to the same pooled vector.
    reg_target: str = "pooled"
    use_ema_target: bool = True       # sigreg arm may set this False (no target network)
    # VICReg
    vicreg_var_gamma: float = 1.0
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 0.04
    # soft vector-quantised codebook
    vq_codebook_size: int = 512
    vq_temperature: float = 0.5
    vq_entropy_weight: float = 0.1
    # SIGReg / LeJEPA
    sigreg_n_slices: int = 64
    sigreg_n_freqs: int = 16


@dataclass
class OptimConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 5e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    ema_start: float = 0.996
    ema_end: float = 1.0
    num_workers: int = 2
    seed: int = 0

    #: Cap the number of optimiser steps per epoch. Without this an epoch is one pass
    #: over the corpus, so attaching a large corpus silently multiplies wall-clock:
    #: ~1900 steps/epoch x 30 epochs x ~0.5 s/step is ~8 h *per model*, and the
    #: notebook trains up to 9. Epochs then stop meaning "a pass over the data" and
    #: start meaning "a fixed compute budget", which is what you actually want when
    #: comparing arms on a fixed session.
    max_steps_per_epoch: Optional[int] = None

    # ---- collapse guard -------------------------------------------------------------
    #: How often (in steps) to test for representation collapse. 0 disables the check.
    collapse_check_every: int = 200
    #: 'stop'   -- abandon this run as soon as collapse is detected (default: a
    #:             collapsed model is worthless, and continuing burns GPU hours)
    #: 'warn'   -- report once and keep going (what RQ3 wants: collapse is the
    #:             measurement, not a failure)
    #: 'ignore' -- no check
    on_collapse: str = "stop"
    #: A latent-prediction loss this far below its own starting value is the trivial
    #: constant solution; legitimate training does not get there in a few hundred steps.
    collapse_loss_frac: float = 0.02
    #: Companion embedding test: near-identical directions for every sample.
    collapse_cos_sim: float = 0.9995


@dataclass
class ForwardSamplingConfig:
    """How (context, future) pairs are drawn from long recordings (Section 5.3b)."""

    context_windows: int = 1          # number of consecutive 10 s windows in the context
    # horizons (seconds) sampled during pretraining; each maps to a horizon-token bucket
    horizons_s: Tuple[float, ...] = (10.0, 30.0, 60.0, 300.0)
    # allow "future" windows drawn from a different recording of the same patient (RQ4)
    patient_aware: bool = False
    patient_aware_prob: float = 0.25
    # Intra-window forward prediction. A 10 s clip has no "next window", so for the
    # short corpora (MIMIC/CODE-15/Chapman/Ningbo/CPSC/PTB-XL) we instead predict the
    # latents of the *later* time patches from the earlier ones inside the same clip.
    # This is the "short intra-recording horizons" de-risking step of Stage 1 and it
    # makes L_fwd trainable on all ~800k short recordings, not only on Holter data.
    intra_window: bool = True
    intra_window_frac: float = 0.6    # fraction of time patches used as context


@dataclass
class HierarchicalConfig:
    """Level-2 (rhythm) stack for RQ2."""

    enabled: bool = False
    l2_embed_dim: int = 384
    l2_depth: int = 4
    l2_num_heads: int = 6
    l2_predictor_dim: int = 192
    l2_predictor_depth: int = 2
    l2_context_windows: int = 12      # 12 x 10 s = 2 min of context
    l2_horizons_s: Tuple[float, ...] = (60.0, 300.0, 600.0, 1800.0)
    l2_weight: float = 1.0
    l2_mask_weight: float = 0.5


@dataclass
class DataConfig:
    input_root: str = "/kaggle/input"
    cache_dir: str = "/kaggle/working/cache"
    # hard caps so a Kaggle session never OOMs on the ~800k-recording corpora
    max_records_per_dataset: int = 20000
    max_long_records: int = 40
    pretrain_datasets: Sequence[str] = (
        "meeti", "mimic", "code15", "chapman", "ningbo", "cpsc", "ptbxl",
    )
    long_datasets: Sequence[str] = ("ltaf", "mitbih")
    # The MIT-BIH *Noise Stress Test* database is deliberately noise-corrupted and
    # ships three pure-noise records (bw/em/ma). Off by default: it is fine as
    # augmentation-like pretraining data but must never be read as an ECG corpus.
    use_noise_corpora: bool = False
    allow_synthetic_fallback: bool = False
    synthetic_n_short: int = 512
    synthetic_n_long: int = 12
    synthetic_long_minutes: float = 40.0


@dataclass
class EvalConfig:
    linear_probe_epochs: int = 40
    linear_probe_lr: float = 1e-3
    linear_probe_batch_size: int = 128
    bootstrap_n: int = 2000
    label_fractions: Tuple[float, ...] = (0.01, 0.1, 0.5, 1.0)
    # anticipation benchmark
    anticipation_horizons_s: Tuple[float, ...] = (300.0, 900.0, 1800.0)
    anticipation_context_s: float = 10.0
    anticipation_stride_s: float = 30.0
    # exclusion margin: windows within this many seconds *before* an event are dropped
    # from the negative pool to avoid label leakage on the boundary
    anticipation_guard_s: float = 5.0


@dataclass
class ExperimentConfig:
    name: str = "cardio_jepa"
    signal: SignalConfig = field(default_factory=SignalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    forward: ForwardSamplingConfig = field(default_factory=ForwardSamplingConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    out_dir: str = "/kaggle/working/runs"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------------------
#: Per-arm regulariser strengths for RQ3.
#:
#: `vicreg` beta was measured on the failing case (forward-only, which collapses
#: outright under EMA-only). Sweeping beta on the *pooled* target:
#:
#:     beta    pooled std   cos_sim   L_reg share of total loss
#:     ema     0.00057      1.00000   -           <- collapsed
#:     0.1     0.38693      0.309     22%
#:     1.0     0.39229      0.636     55%
#:     5.0     0.38802      0.174     68%
#:     20.0    0.40832      0.203     69%
#:
#: The variance floor saturates by 0.1 (0.387 of an eventual 0.408); larger beta buys
#: almost no extra spread and progressively replaces the prediction objective, which
#: is the regime where downstream accuracy degrades. Hence 0.1.
#:
#: Caveat worth knowing: this was chosen on embedding geometry and loss balance, NOT
#: on downstream accuracy. A linear probe over beta in {0.1, 1.0, 5.0} on the
#: synthetic AF task returned 0.855 / 0.842 / 0.856 -- and the *collapsed* EMA model
#: scored 0.873, the highest of all. That task separates on heart rate alone, so a
#: near-degenerate representation still carries enough residual to win it; the probe
#: simply cannot rank representation quality here. Re-validate beta on PTB-XL or
#: CPSC2018 before treating it as tuned.
ANTI_COLLAPSE_DEFAULTS = {
    "ema":    {"beta_reg": 0.0,  "use_ema_target": True},
    "vicreg": {"beta_reg": 0.1, "use_ema_target": True},
    "vq":     {"beta_reg": 0.5,  "use_ema_target": True},
    "sigreg": {"beta_reg": 0.5,  "use_ema_target": False},
}


def configure_arm(cfg: "ExperimentConfig", arm: str) -> "ExperimentConfig":
    """Return a copy of ``cfg`` set up for one RQ3 anti-collapse arm."""
    import copy as _copy
    if arm not in ANTI_COLLAPSE_DEFAULTS:
        raise ValueError(f"unknown arm {arm!r}; expected one of "
                         f"{sorted(ANTI_COLLAPSE_DEFAULTS)}")
    out = _copy.deepcopy(cfg)
    out.name = f"{cfg.name}_{arm}"
    out.loss.anti_collapse = arm
    for k, v in ANTI_COLLAPSE_DEFAULTS[arm].items():
        setattr(out.loss, k, v)
    return out


def smoke_config() -> ExperimentConfig:
    """Tiny end-to-end run: verifies every code path in a couple of minutes on CPU."""
    cfg = ExperimentConfig(name="smoke")
    cfg.model.encoder_embed_dim = 96
    cfg.model.encoder_depth = 2
    cfg.model.encoder_num_heads = 4
    cfg.model.predictor_embed_dim = 64
    cfg.model.predictor_depth = 2
    cfg.model.predictor_num_heads = 4
    cfg.optim.epochs = 2
    cfg.optim.batch_size = 4
    cfg.optim.warmup_epochs = 0
    cfg.optim.amp = False
    cfg.optim.num_workers = 0
    cfg.data.max_records_per_dataset = 64
    cfg.data.max_long_records = 6
    cfg.data.synthetic_n_short = 64
    cfg.data.synthetic_n_long = 6
    cfg.data.synthetic_long_minutes = 30.0
    cfg.evaluation.linear_probe_epochs = 3
    cfg.evaluation.bootstrap_n = 100
    cfg.hierarchical.l2_context_windows = 4
    cfg.out_dir = "/tmp/cardio_jepa_runs"
    return cfg


def kaggle_config() -> ExperimentConfig:
    """Fits a P100 (16 GB) / 2xT4 Kaggle session with time to spare for evaluation."""
    cfg = ExperimentConfig(name="cardio_jepa_kaggle")
    cfg.model.encoder_embed_dim = 384
    cfg.model.encoder_depth = 8
    cfg.model.encoder_num_heads = 6
    cfg.model.predictor_embed_dim = 192
    cfg.model.predictor_depth = 4
    cfg.optim.epochs = 30
    cfg.optim.batch_size = 32
    cfg.optim.lr = 5e-4
    # An epoch here is a fixed compute budget, not a pass over the corpus. With a real
    # corpus attached a full pass is ~1900 steps, so 30 unbounded epochs is ~8 h for
    # ONE model and the notebook trains up to nine. 250 x 30 = 7500 steps/model keeps
    # the default all-stages run inside a single Kaggle session; raise it if you run
    # fewer stages. The estimator cell in the notebook computes the value for the
    # machine you actually get.
    cfg.optim.max_steps_per_epoch = 250
    # L_fwd has a trivial constant solution, so the pooled representation needs an
    # explicit variance floor -- see ANTI_COLLAPSE_DEFAULTS and Stage 3.
    cfg.loss.anti_collapse = "vicreg"
    cfg.loss.reg_target = "pooled"
    cfg.loss.beta_reg = ANTI_COLLAPSE_DEFAULTS["vicreg"]["beta_reg"]
    cfg.data.max_records_per_dataset = 20000
    # LTAFDB ships 84 records and MIT-BIH Arrhythmia 48; the cap applies per corpus,
    # so 100 takes all of both. Long records cost almost nothing to hold -- only
    # annotations live in RAM, signals are read from disk per window (~145 MB for all
    # 84 LTAFDB beat-annotation sets) -- and the anticipation benchmark is
    # patient-level, so every extra record is an extra independent split unit.
    cfg.data.max_long_records = 100
    return cfg


def full_config() -> ExperimentConfig:
    """Paper-scale settings (multi-GPU / multi-day; will not finish in one Kaggle run)."""
    cfg = ExperimentConfig(name="cardio_jepa_full")
    cfg.model.encoder_embed_dim = 768
    cfg.model.encoder_depth = 12
    cfg.model.encoder_num_heads = 16
    cfg.model.predictor_embed_dim = 384
    cfg.model.predictor_depth = 6
    cfg.model.predictor_num_heads = 12
    cfg.model.drop_path_rate = 0.1
    cfg.optim.epochs = 100
    cfg.optim.batch_size = 64
    cfg.optim.lr = 5e-5
    cfg.data.max_records_per_dataset = 10 ** 9
    cfg.data.max_long_records = 10 ** 9
    return cfg
