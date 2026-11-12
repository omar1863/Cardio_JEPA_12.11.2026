"""Cardio-JEPA: latent forward-prediction JEPA for cardiac world modelling from ECG.

Reference implementation of the research proposal
"Cardio-JEPA: Latent Forward-Prediction Joint-Embedding Architecture for Cardiac
World Modeling from ECG".

Built on top of, and attributing, these public codebases:
  * Kim, ECG-JEPA (arXiv:2410.08559)  https://github.com/sehunfromdaegu/ECG_JEPA
      -> CroPA encoder, masked lead-wise predictor, sin-cos 2-D positions,
         EMA target schedule, linear-probe protocol.
  * Assran et al., I-JEPA (CVPR 2023) https://github.com/facebookresearch/ijepa
      -> ViT blocks, context/target asymmetry, predictor design.
  * Na et al., ST-MEM (ICLR 2024)     -> masked raw-reconstruction baseline.

New here (the proposal's contributions): the forward latent predictor with
horizon tokens, the two-level hierarchical rhythm stack, the four-arm
anti-collapse study, and the anticipation benchmarks.
"""

from .config import (ANTI_COLLAPSE_DEFAULTS, DataConfig, EvalConfig, ExperimentConfig,
                     ForwardSamplingConfig, HierarchicalConfig, HORIZON_BUCKETS_S,
                     LossConfig, ModelConfig, OptimConfig, REDUCED_LEAD_IDX,
                     REDUCED_LEAD_NAMES, SignalConfig, configure_arm, full_config,
                     kaggle_config, smoke_config)
from .checkpoint import (describe_model_input, find_checkpoints, find_package_root,
                         load_weights)
from .diagnostics import CollapseMonitor, detect_collapse_signature, effective_rank
from .engine import InterleavedLoader, pick_device, pretrain, set_seed
from .models.baselines import Inception1d, STMEMBaseline, XResNet1d
from .models.cardio_jepa import CardioJEPA

__version__ = "0.1.0"

__all__ = [
    "CardioJEPA", "STMEMBaseline", "XResNet1d", "Inception1d",
    "ExperimentConfig", "SignalConfig", "ModelConfig", "LossConfig", "OptimConfig",
    "ForwardSamplingConfig", "HierarchicalConfig", "DataConfig", "EvalConfig",
    "HORIZON_BUCKETS_S", "REDUCED_LEAD_IDX", "REDUCED_LEAD_NAMES",
    "smoke_config", "kaggle_config", "full_config", "configure_arm",
    "ANTI_COLLAPSE_DEFAULTS",
    "pretrain", "pick_device", "set_seed", "InterleavedLoader",
    "CollapseMonitor", "effective_rank", "detect_collapse_signature",
    "load_weights", "find_package_root", "find_checkpoints", "describe_model_input",
    "__version__",
]
