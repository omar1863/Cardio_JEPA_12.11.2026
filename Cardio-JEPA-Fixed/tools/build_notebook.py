#!/usr/bin/env python3
"""Generate the Kaggle notebook.

The notebook imports `cardio_jepa` from a Kaggle **Model** input rather than writing
the source into itself, so this script only emits narrative and experiment-driver
cells. Produce the model input with `tools/export_kaggle_model.py`.

Usage:  python tools/build_notebook.py [-o notebooks/cardio_jepa_kaggle.ipynb]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CELL_N = [0]


def _next_id() -> str:
    """nbformat >= 4.5 requires a unique id on every cell."""
    _CELL_N[0] += 1
    return f"cell-{_CELL_N[0]:03d}"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {},
            "source": src.splitlines(True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "id": _next_id(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": src.rstrip("\n").splitlines(True)}


# ======================================================================================
# narrative + driver cells
# ======================================================================================
HEADER = r"""# Cardio-JEPA — Latent Forward-Prediction JEPA for Cardiac World Modelling from ECG

Reference implementation of the research proposal *"Cardio-JEPA: Latent
Forward-Prediction Joint-Embedding Architecture for Cardiac World Modeling from ECG"*,
packaged to run start-to-finish inside one Kaggle session.

## What is new here

All three published ECG-JEPAs — Kim's ECG-JEPA (arXiv:2410.08559), Weimann & Conrad's
I-JEPA-for-ECG (arXiv:2410.13867), and the Hierarchical/ER-JEPA (arXiv:2607.01145) —
predict **masked latent patches inside a single static ~10 s recording**. None predicts
*forward in time*. Cardio-JEPA adds that:

$$\mathcal{L} \;=\; \underbrace{w_{\text{fwd}}\,\mathcal{L}_{fwd}}_{\text{dynamics (new)}}
\;+\; \underbrace{\lambda\,\mathcal{L}_{mask}}_{\text{morphology (Kim)}}
\;+\; \underbrace{\beta\,\mathcal{L}_{reg}}_{\text{anti-collapse (RQ3)}}$$

$$\mathcal{L}_{fwd}=\big\lVert g_\phi\big(f_\theta(x_{[t-L,t]}),\,h\big)-\mathrm{sg}\big(\bar f_\theta(x_{[t,t+H]})\big)\big\rVert$$

where $h$ is a learned **horizon token** so one model can be queried at 2 s … 1 h.

## Code reuse and attribution

| Source | Repository | What is reused |
|---|---|---|
| Kim, ECG-JEPA (2024) | `sehunfromdaegu/ECG_JEPA` | Cross-Pattern Attention encoder, lead-wise masked predictor, 2-D sin-cos positions, EMA schedule, linear-probe protocol |
| Assran et al., I-JEPA (CVPR 2023) | `facebookresearch/ijepa` | ViT blocks, context/target asymmetry, predictor design |
| Na et al., ST-MEM (ICLR 2024) | — | masked raw-reconstruction baseline |

Written new for this proposal: the forward predictor with horizon tokens, the two-level
hierarchical rhythm stack, the four-arm anti-collapse study, the anticipation benchmarks,
and a dependency-free WFDB reader.

## How the notebook is organised

1. **Setup** — import `cardio_jepa` from an attached Kaggle Model input, and
   optionally warm-start from a checkpoint bundled with it.
2. **Data** — auto-discover whatever ECG corpora are attached; fall back to synthetic.
3. **Stage 1 (de-risk)** — masked-only (Kim reproduction) vs forward-only vs dual, on the
   standard PTB-XL / CPSC2018 linear probe.
4. **Stage 2 (H1)** — near-term rhythm-onset anticipation, with label-efficiency curves
   and paired bootstrap tests.
5. **Stage 3 (RQ3)** — the four anti-collapse arms with effective-rank diagnostics.
6. **Stage 4 (H2)** — hierarchical vs single-scale across increasing horizons.
7. **Summary** — every table written to `/kaggle/working/results/`.

> **Run modes.** Set `RUN_MODE` in the config cell. `smoke` finishes in a few minutes
> (CPU is fine) and exercises every code path; `kaggle` is sized for one GPU session;
> `full` is the paper-scale configuration and will not finish in a single session.
"""

DATA_MD = r"""## Attaching the datasets

Use **Add Input → Datasets**. Discovery walks `/kaggle/input` and matches *directory
names at any depth*, so both mount layouts work:

```
/kaggle/input/<slug>/...                            (classic)
/kaggle/input/datasets/<owner>/<slug>/<subdir>/...  (current)
```

Nothing below is required; anything missing is skipped.

| Role in the proposal | Kaggle dataset | Size |
|---|---|---|
| MIMIC-IV-ECG — primary large-scale pretraining | `ishtiaqueahmed354/metti-dataset` (**MEETI**) | varies |
| MIMIC-IV-ECG (raw mirrors) | `biswajitkrshaw/mimic-iv-ecg-diagnostic-electrocardiogram` · `beckpro/mimic-iv-ecg-wave-npy-dataset` | 38.5 / 1.7 GB |
| CODE-15 — pretraining | `gerardolug/ecg-code-15` · `ccf002/code-15-ecg-complete` | 5–46 GB |
| Chapman-Shaoxing + Ningbo | `bjoernjostein/shaoxing-and-ningbo-first-hospital-database` | 2.5 GB |
| Chapman-Shaoxing (alone) | `erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database` | 0.6 GB |
| CPSC2018 — pretraining + downstream | `physionet/china-physiological-signal-challenge-in-2018` | 0.9 GB |
| PTB-XL — pretraining + primary benchmark | `khyeh0719/ptb-xl-dataset` · `omar1863/ptb-xl` | 1.9 GB |
| **Long-Term AF — required for Stages 2 & 4** | `sahilshahare12/long-term-af-database` | 1.8 GB |
| **MIT-BIH Arrhythmia — required for Stages 2 & 4** | `protobioengineering/mit-bih-arrhythmia-database-modern-2023` | 0.2 GB |

**MEETI** is derived from MIMIC-IV-ECG (784,680 12-lead 10 s ECGs from 160,597 ICU
patients) and fills the proposal's primary pretraining slot on its own.

Attaching every mirror at once is not advisable — Kaggle limits attached input and
writable output, and indexing a 38 GB mount costs real minutes before training starts.
Where two mounts resolve to the same directory (a combined Chapman+Ningbo mirror serves
both corpus keys), it is indexed once; otherwise every record would be loaded twice.

### Why the objective arms all carry a variance floor

`L_fwd` has an exact trivial solution: if the encoder emits one constant vector, the
LayerNormed target is a fixed vector and the predictor just memorises it, driving the
loss to zero while the representation carries nothing. This is not hypothetical — under
EMA-only it happens within a few hundred steps, loss `0.39 -> 0.0001`, effective rank
`1.0`.

So all Stage 1 arms use the same anti-collapse setting (`vicreg` on the **pooled**
representation) rather than pure EMA. That keeps the objective comparison controlled —
every arm differs only in `w_fwd` and `lambda` — at the cost that `masked_only` is then
"Kim's objective plus a variance floor" rather than a bit-exact reproduction.

Pure EMA-only behaviour is not lost: it is precisely what **Stage 3** measures, which is
where the proposal's RQ3 lives.

> **`beta` is not yet validated downstream.** It was set from embedding geometry (the
> variance floor saturates at 0.1; larger values progressively replace the prediction
> objective). A linear probe could not rank it: on the synthetic AF task, β ∈ {0.1, 1,
> 5} scored 0.855 / 0.842 / 0.856 — and the *collapsed* EMA model scored **0.873**, the
> best of all. That task separates on heart rate alone, so even a degenerate
> representation wins it. Treat any AUROC ordering from synthetic data as uninformative
> about representation quality, and re-check β on PTB-XL or CPSC2018.

### Two corpora that are easy to mistake for each other

The **MIT-BIH Noise Stress Test** database (`mit-bih-noise-stress-test-database`) is
*not* the MIT-BIH Arrhythmia Database. It is 12 deliberately noise-corrupted records
plus three **pure-noise** records (`bw`, `em`, `ma` — baseline wander, electrode motion,
muscle artefact), and it carries no AF rhythm annotations. Discovery classifies it
separately, excludes the three noise-only records, and leaves it out of pretraining
unless you set `CFG.data.use_noise_corpora = True`. It is never used as an anticipation
corpus.

### On the two "required" corpora

`L_fwd` needs a *future* window. A 10 s clip does not have one, so:

* **long recordings** (LTAFDB ~24 h, MIT-BIH 30 min) drive genuine cross-window
  forecasting at horizons of seconds to tens of minutes;
* **10 s clips** drive an *intra-window causal* variant — context is the first 60 % of
  time patches, the target is the later patches, which the encoder never sees. That is
  the "short intra-recording horizons" de-risking step from Stage 1 of the proposal, and
  it makes `L_fwd` trainable on all ~800k short recordings rather than only on Holter data.

Both paths share the same predictor and the same horizon-token machinery.

**If neither LTAFDB nor MIT-BIH Arrhythmia is attached**, discovery says so explicitly
and: `L_fwd` trains only via the intra-window path, and Stages 2 and 4 fall back to
synthetic recordings — which validates the pipeline but says nothing about ECG. Stages 1
and 3 are unaffected and still produce real results from the short corpora.
"""

SETUP_CELL = r'''# --- environment -------------------------------------------------------------------
import os, sys, json, time, math, warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.chdir(WORK)
for d in ("results", "runs"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)

import torch
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__)
print("cuda    ", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU only)")
print("workdir ", WORK)

# Optional: `wfdb` is used when present, but the bundled reader covers every format
# these databases use, so a missing package (or disabled internet) is not a problem.
try:
    import wfdb  # noqa: F401
    print("wfdb     available (will be preferred)")
except ImportError:
    print("wfdb     not installed - using the bundled dependency-free reader")
'''

BOOTSTRAP_MD = r"""### Locating the `cardio_jepa` package

The package is loaded from a **Kaggle Model input**, not written out by the notebook.
Attach it via **Add Input → Models**; the default path below is the first place we look.

To produce that model input from the repository:

```bash
python tools/export_kaggle_model.py --checkpoint runs/dual.pth
# then upload build/kaggle_model/ at https://www.kaggle.com/models -> New Model
#   Framework=PyTorch   Variation=default   Version=1   slug=cardio-jepa
```

The search below falls back to `/kaggle/input/models`, then all of `/kaggle/input`,
then `/kaggle/working` — so a package uploaded as an ordinary *dataset*, or left over
in the working directory from an earlier session, is also found. The path-finding code
is duplicated here rather than imported because it has to run *before* the package is
importable.
"""

BOOTSTRAP_CELL = r'''# --- locate and import the cardio_jepa package --------------------------------------
MODEL_INPUT = "/kaggle/input/models/ishtiaqueahmed354/cardio-jepa/pytorch/default/1"

PACKAGE_SEARCH_ROOTS = [
    MODEL_INPUT,              # the exact path, when it is where we expect
    "/kaggle/input/models",   # any other model input
    "/kaggle/input",          # the package uploaded as a plain dataset
    WORK,                     # left over in /kaggle/working from an earlier session
]


def find_package_root(roots, max_depth=8):
    """Return a directory D with D/cardio_jepa/__init__.py, or None."""
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        stack = [(root, 0)]
        while stack:
            path, depth = stack.pop()
            if depth > max_depth:
                continue
            if os.path.isfile(os.path.join(path, "cardio_jepa", "__init__.py")):
                return path
            try:
                for e in os.scandir(path):
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                        stack.append((e.path, depth + 1))
            except (PermissionError, FileNotFoundError, NotADirectoryError):
                continue
    return None


from pathlib import Path
_fixed = sorted(Path("/kaggle/input").rglob("FIX_REPORT.md")) if Path("/kaggle/input").exists() else []
PKG_ROOT = find_package_root([str(f.parent) for f in _fixed] + PACKAGE_SEARCH_ROOTS)

if PKG_ROOT is None:
    print("Searched for cardio_jepa/__init__.py under:")
    for r in PACKAGE_SEARCH_ROOTS:
        print(f"  {'[exists]' if os.path.isdir(r) else '[missing]'} {r}")
    if os.path.isdir("/kaggle/input/models"):
        print("\nModel inputs currently attached:")
        for dirpath, dirnames, _f in os.walk("/kaggle/input/models"):
            if dirpath.count(os.sep) - "/kaggle/input/models".count(os.sep) < 4:
                for d in dirnames:
                    print("   ", os.path.join(dirpath, d))
    raise FileNotFoundError(
        "cardio_jepa package not found.\n"
        "Fix: run `python tools/export_kaggle_model.py` in the repo, upload "
        "build/kaggle_model/ as a Kaggle Model (Framework=PyTorch, Variation=default, "
        "Version=1), attach it to this notebook, and set MODEL_INPUT above to its "
        "mount path.")

sys.path.insert(0, PKG_ROOT)
for _m in [m for m in list(sys.modules) if m == "cardio_jepa" or m.startswith("cardio_jepa.")]:
    del sys.modules[_m]

import cardio_jepa as cj
print(f"cardio_jepa {cj.__version__} loaded from {os.path.dirname(cj.__file__)}")
'''

WARMSTART_MD = r"""### Warm-starting from a previous run

Any checkpoint bundled with the model input (or left in `/kaggle/working/runs/`) can
seed this run instead of starting from random init. Set `WARM_START` below.

`load_weights` is deliberately non-strict: resuming at a *different* encoder width is a
normal thing to want, and it legitimately restores only the tensors that still fit. It
reports the fraction restored and warns when that drops under half — silently loading
3% of a model and calling it a warm start would be worse than saying so.

`"auto"` searches in **priority order** — the model input first, then any other model
input, then `/kaggle/working/runs/` — and takes the newest checkpoint in the
highest-priority root that has one. Order, not file size, decides: a scratch checkpoint
left in `runs/` by an earlier session must never outrank the one you deliberately
shipped with the model.
"""

WARMSTART_CELL = r'''# --- optional warm start ------------------------------------------------------------
WARM_START = "off"      # "auto" | "off" | an explicit path to a .pth

print(cj.describe_model_input(MODEL_INPUT) if os.path.isdir(MODEL_INPUT)
      else f"{MODEL_INPUT}: not mounted")

WARM_START_PATH = None
if WARM_START == "auto":
    cands = cj.find_checkpoints([MODEL_INPUT, "/kaggle/input/models",
                                 os.path.join(WORK, "runs")])
    WARM_START_PATH = cands[0] if cands else None
    print(f"\nauto warm start: {WARM_START_PATH or 'no checkpoint found (fresh init)'}")
elif WARM_START != "off":
    WARM_START_PATH = WARM_START

if WARM_START_PATH and not os.path.isfile(WARM_START_PATH):
    print(f">>> {WARM_START_PATH} does not exist - starting from scratch instead")
    WARM_START_PATH = None


def maybe_warm_start(model, name=""):
    """Apply the warm start, if one is configured. Returns the load report or None."""
    if not WARM_START_PATH:
        return None
    rep = cj.load_weights(model, WARM_START_PATH, strict=False, verbose=False)
    print(f"    warm start [{name}]: {rep.summary().splitlines()[0]}")
    if rep.shape_mismatch:
        print(f"      {len(rep.shape_mismatch)} tensors skipped on shape mismatch")
    return rep
'''

IMPORT_CELL = r'''# --- pull in the pieces the experiment cells use ------------------------------------
from cardio_jepa.config import configure_arm, full_config, kaggle_config, smoke_config
from cardio_jepa.data import (build_arrhythmia_onset_dataset, build_burden_dataset,
                              build_onset_dataset, discover, split_by_record, summarize)
from cardio_jepa.data.build import (build_pretrain_loaders, build_probe_splits,
                                    load_long_corpus, probe_batch_from_loader)
from cardio_jepa.eval import (bootstrap_auroc_delta, compare_models, forecast_skill,
                              horizon_sweep, label_efficiency_table,
                              run_anticipation_probe, run_burden_probe,
                              run_classification_probe)
from cardio_jepa.models import CardioJEPA, STMEMBaseline

import pandas as pd
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

DEVICE = cj.pick_device()
print("cardio_jepa", cj.__version__, "| device:", DEVICE)
'''

DISCOVER_CELL = r'''# --- what is actually attached? -----------------------------------------------------
INPUT_ROOT = "/kaggle/input"
FOUND = discover(INPUT_ROOT, verbose=True)
print()
print(summarize(FOUND))
if not FOUND:
    print("\n>>> No corpora detected. The notebook will run on synthetic ECG so every "
          "cell still executes -- treat the numbers as a pipeline smoke test, not "
          "as evidence about ECG.")
'''

CONFIG_MD = r"""## Configuration

`RUN_MODE` is the one knob that matters.

| mode | encoder | epochs | typical wall clock | use for |
|---|---|---|---|---|
| `smoke` | 96-d, 2 layers | 2 | minutes, CPU-ok | verifying every path runs |
| `kaggle` | 384-d, 8 layers | 30 | a few hours on P100/T4 | real results in one session |
| `full` | 768-d, 12 layers | 100 | days, multi-GPU | the paper-scale run |

`STAGES` selects which experiments to run — each stage trains its own models, so
dropping stages is the main way to fit the GPU budget.

**Budget.** With all stages on, the notebook pretrains **9 models**: 4 in Stage 1
(masked-only, forward-only, dual, ST-MEM), 3 more in Stage 3 (the `ema` arm reuses
Stage 1's `dual`, since they are the same run), and 2 in Stage 4. In `kaggle` mode that
is 9 × 30 epochs, so check the per-epoch time printed by Stage 1 and cut `STAGES` or
`CFG.optim.epochs` if it will not fit your session. Every stage saves checkpoints to
`/kaggle/working/runs/`, so stages can also be run across several sessions.
"""

CONFIG_CELL = r'''# --- configuration ------------------------------------------------------------------
RUN_MODE = "smoke"          # "smoke" | "kaggle" | "full"
STAGES   = {
    "stage1_objective_ablation": True,   # masked-only vs forward-only vs dual
    "stage2_anticipation":       True,   # H1
    "stage3_collapse":           False,   # RQ3
    "stage4_hierarchical":       False,   # H2
}

CFG = {"smoke": smoke_config, "kaggle": kaggle_config, "full": full_config}[RUN_MODE]()
CFG.data.input_root = INPUT_ROOT
CFG.data.allow_synthetic_fallback = (RUN_MODE == "smoke")
CFG.out_dir = os.path.join(WORK, "runs")
CFG.optim.num_workers = 2 if RUN_MODE != "smoke" else 0
CFG.forward.context_windows = 2          # 20 s of context for the forward objective
CFG.hierarchical.enabled = False         # switched on per-stage below

RESULTS_DIR = os.path.join(WORK, "results")
RESULTS = {}

cj.set_seed(CFG.optim.seed)
print(f"mode={RUN_MODE}  encoder={CFG.model.encoder_embed_dim}d x {CFG.model.encoder_depth}L  "
      f"epochs={CFG.optim.epochs}  batch={CFG.optim.batch_size}")
print(f"window: {CFG.signal.window_s}s @ {CFG.signal.fs}Hz -> "
      f"{CFG.signal.n_leads} leads x {CFG.signal.n_time_patches} patches "
      f"= {CFG.signal.n_tokens} tokens")
print("horizon buckets (s):", list(cj.HORIZON_BUCKETS_S))
'''

DATA_BUILD_CELL = r'''# --- build loaders and probe splits -------------------------------------------------
t0 = time.time()
LONG_RECORDS = load_long_corpus(FOUND, CFG, verbose=True)
_rng = np.random.default_rng(CFG.optim.seed)
_n_test = max(1, round(len(LONG_RECORDS)*0.3)) if len(LONG_RECORDS) >= 2 else 0
HELD_OUT_RECORDS = set(_rng.permutation(len(LONG_RECORDS))[:_n_test].tolist())
CFG.evaluation.held_out_records = sorted(HELD_OUT_RECORDS)
PRETRAIN_LONG_RECORDS = [r for i, r in enumerate(LONG_RECORDS) if i not in HELD_OUT_RECORDS]
print("Long-record pretraining/evaluation holdout:", len(PRETRAIN_LONG_RECORDS), len(HELD_OUT_RECORDS))
LOADER, DATA_INFO = build_pretrain_loaders(FOUND, CFG, long_records=PRETRAIN_LONG_RECORDS,
                                           samples_per_record=200 if RUN_MODE != "smoke" else 32,
                                           verbose=True)
PROBE_BATCH = probe_batch_from_loader(LOADER, n=128 if RUN_MODE != "smoke" else 32)
PROBE_SPLITS = build_probe_splits(FOUND, CFG, verbose=True)

print(f"\nready in {time.time() - t0:.1f}s")
print(f"  short clips     : {DATA_INFO.get('n_short', 0)}")
print(f"  forward pairs   : {DATA_INFO.get('n_forward_pairs', 0)}")
print(f"  long recordings : {len(LONG_RECORDS)} "
      f"({sum(r.duration_s for r in LONG_RECORDS) / 3600:.1f} h total)")
print(f"  probe tasks     : {list(PROBE_SPLITS)}")
print(f"  steps/epoch     : {DATA_INFO['steps_per_epoch']}")

n_rhythm = sum(1 for r in LONG_RECORDS if r.episodes)
if n_rhythm == 0:
    print("\n>>> No rhythm annotations found: the anticipation benchmarks need `.atr` "
          "rhythm labels (LTAFDB / MIT-BIH). Stage 2 will be skipped.")
else:
    burdens = [r.burden() for r in LONG_RECORDS if r.episodes]
    print(f"  AF burden       : mean {np.mean(burdens):.3f} over {n_rhythm} records")
'''

BUDGET_MD = r"""### Runtime budget — check this before committing a session

An epoch is a **fixed step budget** (`optim.max_steps_per_epoch`), not a pass over the
corpus. That matters: with a real corpus attached a full pass is ~1900 steps, so 30
unbounded epochs is ~8 hours for *one* model, and the default stages train nine. The
first symptom of getting this wrong is discovering it eight hours in.

The cell below times ten real steps on the machine you actually got and projects the
total. If the projection exceeds the session budget it prints the `max_steps_per_epoch`
that would fit.
"""

BUDGET_CELL = r'''# --- runtime projection --------------------------------------------
from cardio_jepa.engine import budget_steps_per_epoch, estimate_runtime, report_runtime

BUDGET_HOURS = 11.0        # Kaggle sessions are capped around 12 h

# how many models the enabled stages will train
N_MODELS = (4 if STAGES["stage1_objective_ablation"] else 0) \
         + (4 if STAGES["stage3_collapse"] else 0) \
         + (2 if STAGES["stage4_hierarchical"] else 0)
N_MODELS = max(N_MODELS, 1)

_probe_model = CardioJEPA(CFG.signal, CFG.model, CFG.loss, CFG.hierarchical, CFG.forward)
EST = estimate_runtime(_probe_model, LOADER, CFG, DEVICE, n_steps=10, n_models=N_MODELS)
del _probe_model

print(f"{N_MODELS} model(s) to train, {CFG.optim.epochs} epochs, "
      f"max_steps_per_epoch={CFG.optim.max_steps_per_epoch}")
print(report_runtime(EST, budget_hours=BUDGET_HOURS))

if EST["hours_total"] > BUDGET_HOURS:
    fits = budget_steps_per_epoch(EST["s_per_step"], CFG.optim.epochs, N_MODELS,
                                  BUDGET_HOURS)
    print(f"\n  Suggested: CFG.optim.max_steps_per_epoch = {fits}")
    print(f"  Applying it now; comment out the next line to keep your own value.")
    CFG.optim.max_steps_per_epoch = fits
    EST = estimate_runtime(CardioJEPA(CFG.signal, CFG.model, CFG.loss,
                                      CFG.hierarchical, CFG.forward),
                           LOADER, CFG, DEVICE, n_steps=4, n_models=N_MODELS)
    print(report_runtime(EST, budget_hours=BUDGET_HOURS))

RESULTS["runtime_estimate"] = dict(EST)
'''

STAGE1_MD = r"""## Stage 1 — De-risk: does adding $\mathcal{L}_{fwd}$ cost anything on standard classification?

Three architecturally **identical** models, differing only in the objective:

| arm | $w_{fwd}$ | $\lambda$ | what it is |
|---|---|---|---|
| `masked_only` | 0 | 1 | Kim's ECG-JEPA, reproduced |
| `forward_only` | 1 | 0 | the new objective alone |
| `dual` | 1 | 1 | **Cardio-JEPA** |
| `st_mem` | — | — | raw-signal reconstruction baseline |

Go/no-go from the proposal: *the dual objective must match the masked-only baseline on
standard classification (within CI) before proceeding.* Linear probing on frozen
encoders, macro-AUROC, with paired bootstrap CIs.
"""

STAGE1_CELL = r'''# --- Stage 1: objective ablation ----------------------------------------------------
import copy

def build_objective_arm(name):
    cfg = copy.deepcopy(CFG)
    cfg.name = f"{CFG.name}_{name}"
    if name == "masked_only":
        cfg.loss.weight_forward, cfg.loss.lambda_mask = 0.0, 1.0
    elif name == "forward_only":
        cfg.loss.weight_forward, cfg.loss.lambda_mask = 1.0, 0.0
    elif name == "dual":
        cfg.loss.weight_forward, cfg.loss.lambda_mask = 1.0, 1.0
    return cfg


def train_one(name, cfg, model=None):
    cj.set_seed(cfg.optim.seed)
    print(f"\n=== training [{name}] ===")
    if model is None:
        model = CardioJEPA(cfg.signal, cfg.model, cfg.loss, cfg.hierarchical, cfg.forward)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"    {n_par:.1f}M trainable parameters")
    maybe_warm_start(model, name)
    res = cj.pretrain(model, LOADER, cfg, device=DEVICE, arm_name=name,
                      probe_batch=PROBE_BATCH,
                      save_path=os.path.join(cfg.out_dir, f"{name}.pth"),
                      log_every=0 if RUN_MODE == "smoke" else 100)
    if res.collapsed:
        print(f"    [{name}] COLLAPSED at step {res.collapse_step}: {res.collapse_reason}")
        print(f"    Its probe numbers below are meaningless - a collapsed encoder "
              f"carries no signal for any linear head.")
    model.cpu()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model, res


STAGE1 = {}
if STAGES["stage1_objective_ablation"]:
    for arm in ("masked_only", "forward_only", "dual"):
        cfg = build_objective_arm(arm)
        model, res = train_one(arm, cfg)
        STAGE1[arm] = {"model": model, "res": res, "cfg": cfg}

    # ST-MEM style raw-reconstruction baseline (same encoder, different target space)
    stmem_cfg = copy.deepcopy(CFG)
    stmem = STMEMBaseline(stmem_cfg.signal, stmem_cfg.model)
    _, stmem_res = train_one("st_mem", stmem_cfg, model=stmem)
    STAGE1["st_mem"] = {"model": stmem, "res": stmem_res, "cfg": stmem_cfg}
    print("\nStage 1 training complete.")
else:
    print("Stage 1 skipped.")
'''

STAGE1_EVAL_CELL = r'''# --- Stage 1: linear probing on the standard benchmarks -----------------------------
stage1_rows, stage1_scores, stage1_frac = [], {}, {}

if STAGE1 and PROBE_SPLITS:
    for task_name, splits in PROBE_SPLITS.items():
        print(f"\n### linear probe: {task_name} "
              f"({splits['n_classes']} classes, {len(splits['train'])} train)")
        task_scores = {}
        for arm, entry in STAGE1.items():
            print(f"  [{arm}]")
            out = run_classification_probe(entry["model"], splits, DEVICE, CFG.evaluation,
                                           batch_size=64, num_workers=0, verbose=False)
            entry["model"].cpu()
            stage1_rows.append({"task": task_name, "arm": arm,
                                "macro_auroc": out["macro_auroc"],
                                "macro_f1": out["by_fraction"][1.0].get("macro_f1"),
                                "feature_dim": out["feature_dim"],
                                "collapsed": bool(entry["res"].collapsed)})
            task_scores[arm] = out["y_score"]
            stage1_frac[f"{task_name}/{arm}"] = {
                f: r["macro_auroc"] for f, r in out["by_fraction"].items()}
            print(f"    macro AUROC = {out['macro_auroc']:.4f}")
        stage1_scores[task_name] = (splits, task_scores,
                                    out["y_true"])   # y_true is shared across arms

    stage1_df = pd.DataFrame(stage1_rows)
    RESULTS["stage1_linear_probe"] = stage1_df
    display(stage1_df.pivot(index="arm", columns="task", values="macro_auroc")
            .round(4).sort_values(by=list(stage1_df.task.unique())[0], ascending=False))

    # go/no-go: dual vs masked-only, paired bootstrap
    for task_name, (_s, scores, y_true) in stage1_scores.items():
        if "dual" in scores and "masked_only" in scores:
            d = bootstrap_auroc_delta(y_true, scores["dual"], scores["masked_only"],
                                      n=CFG.evaluation.bootstrap_n)
            verdict = ("No significant degradation detected; equivalence is not established"
                       if d["delta"] >= 0 or not d["significant"]
                       else "NO-GO - dual is significantly worse")
            print(f"\n[{task_name}] dual - masked_only = {d['delta']:+.4f} "
                  f"[{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p_value']:.3f}")
            print(f"  -> {verdict}")
            RESULTS[f"stage1_gonogo_{task_name}"] = d
else:
    print("Stage 1 evaluation skipped (no trained models or no labelled probe task).")
'''

STAGE2_MD = r"""## Stage 2 — H1: near-term rhythm-onset anticipation

Given a context window **containing no AF at all**, predict whether an AF episode
*starts* within the next $H$ seconds.

Leakage control (proposal Section 9) is enforced in `build_onset_dataset`:

* the context window must not overlap the target rhythm anywhere;
* a guard band after the context is discarded, so a window whose event begins at its
  very edge is dropped rather than counted as a negative;
* a negative requires the **full horizon to be observed** — windows near the end of a
  record cannot be negatives, because absence there is unobserved, not known;
* train/test are split **by record (patient)**, never by window.

Two feature modes are compared, which is where the world-model claim is actually tested:

* `context` — the pooled frozen encoder embedding (available to every method);
* `world` — `[context embedding ‖ predicted future latent at H]`, i.e. the probe also
  sees *where the model thinks the heart is heading*. Only models with a forward
  predictor have this.

> **Scope note.** This is benchmark 3 from Section 6.2 (near-term, within-record
> anticipation). Benchmark 2 — Attia-style *latent AF*, "will this patient be diagnosed
> within 30 days" — needs patient-level longitudinal diagnosis labels that none of the
> public Kaggle mirrors carry. The dataset builder is horizon-agnostic and would accept
> such labels directly, but no such result is produced here, and none should be claimed.
"""

STAGE2_CELL = r'''# --- Stage 2: anticipation (H1) -----------------------------------------------------
ANTICIPATION_H = 300.0 if RUN_MODE == "smoke" else 900.0   # seconds

stage2_df = None
if STAGES["stage2_anticipation"] and STAGE1 and any(r.episodes for r in LONG_RECORDS):
    samples, stats = build_onset_dataset(
        LONG_RECORDS, CFG.signal, horizon_s=ANTICIPATION_H,
        context_windows=CFG.forward.context_windows,
        stride_s=CFG.evaluation.anticipation_stride_s,
        guard_s=CFG.evaluation.anticipation_guard_s,
        max_per_record=400 if RUN_MODE != "smoke" else 60)
    print(f"AF-onset dataset @ H={ANTICIPATION_H:.0f}s: "
          f"{stats['n_positive']:.0f} positive / {stats['n_negative']:.0f} negative "
          f"(prevalence {stats['prevalence']:.3f})")

    tr, te = split_by_record(samples, len(LONG_RECORDS), test_frac=0.3,
                             seed=CFG.optim.seed, held_out_records=HELD_OUT_RECORDS)
    print(f"record-level split: {len(tr)} train / {len(te)} test windows")

    rows, scores, y_ref = [], {}, None
    if stats["n_positive"] >= 10 and stats["n_negative"] >= 10 and tr and te:
        for arm, entry in STAGE1.items():
            has_predictor = arm in ("dual", "forward_only") and hasattr(entry["model"], "forward_predictor")
            for mode in (("context", "world") if has_predictor else ("context",)):
                r = run_anticipation_probe(
                    entry["model"], LONG_RECORDS, tr, te, CFG.signal, DEVICE,
                    CFG.evaluation, context_windows=CFG.forward.context_windows,
                    feature_mode=mode, horizons_s=[ANTICIPATION_H],
                    batch_size=64, verbose=False)
                entry["model"].cpu()
                if np.isnan(r.get("auroc", float("nan"))):
                    print(f"  {arm}/{mode}: {r.get('note', 'no result')}")
                    continue
                name = f"{arm}/{mode}"
                rows.append({"arm": arm, "features": mode, "auroc": r["auroc"],
                             "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                             "n_test": r["n_test"], "prevalence": r["prevalence"]})
                scores[name] = r["y_score"]
                y_ref = r["y_true"]
                stage1_frac[f"anticipation/{name}"] = r["by_fraction"]
                print(f"  {name:26s} AUROC={r['auroc']:.4f} "
                      f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")

        if rows:
            stage2_df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
            RESULTS["stage2_anticipation"] = stage2_df
            display(stage2_df.round(4))

            ref = "masked_only/context"
            if ref in scores and y_ref is not None:
                print(f"\npaired bootstrap vs {ref}:")
                display(compare_models(y_ref, scores, reference=ref,
                                       n=CFG.evaluation.bootstrap_n).round(4))
    else:
        print("Not enough positive/negative windows for a meaningful probe.")
else:
    print("Stage 2 skipped (needs trained models and rhythm-annotated long recordings).")
'''

STAGE2B_CELL = r'''# --- Stage 2b: label-efficiency curves (the second half of H1) -----------------------
if stage1_frac:
    eff = {k: v for k, v in stage1_frac.items() if k.startswith("anticipation/")}
    if not eff:
        eff = stage1_frac
    eff_df = label_efficiency_table(eff).round(4)
    RESULTS["label_efficiency"] = eff_df
    print("macro AUROC by fraction of labels used "
          "(H1 predicts the Cardio-JEPA margin *grows* as labels shrink):")
    display(eff_df)
else:
    print("No label-efficiency results to report.")
'''

STAGE2C_MD = r"""### Stage 2c — is the predictor forecasting, or just copying?

A forward predictor that simply echoes its context would still score well downstream,
because the current cardiac state is itself predictive. So before reading anything into
the anticipation AUROCs, measure the part that is actually *forecasting*:

$$\text{skill}(H)\;=\;\cos\!\big(g_\phi(z_{ctx},h),\,\bar f_\theta(x_{fut})\big)\;-\;\cos\!\big(z_{ctx},\,\bar f_\theta(x_{fut})\big)$$

on batch-centred embeddings (raw pooled latents share a large constant component that
pins every cosine near 1 and hides the comparison). **Skill ≤ 0 at every horizon means
$\mathcal{L}_{fwd}$ bought nothing**, whatever the probe numbers say — this is the check
that can falsify the core claim, so it is reported before the headline result is trusted.
"""

STAGE2C_CELL = r'''# --- Stage 2c: forecast skill vs a persistence baseline -----------------------------
# The load-bearing sanity check for the whole contribution. A forward predictor that
# merely echoes the present would still produce plausible AUROCs downstream, because the
# context embedding alone is informative. `skill` isolates the part that is actually
# forecasting: cos(prediction, true future) - cos(context, true future), on centred
# embeddings. skill <= 0 at every horizon means L_fwd has bought nothing, whatever the
# probe numbers say.
if STAGE1 and LONG_RECORDS:
    skill_rows = []
    HS_SKILL = ((10.0, 60.0, 300.0) if RUN_MODE == "smoke"
                else (10.0, 60.0, 300.0, 900.0, 1800.0))
    for arm in ("dual", "forward_only"):
        entry = STAGE1.get(arm)
        if entry is None or not hasattr(entry["model"], "forward_predictor"):
            continue
        print(f"[{arm}]")
        df = forecast_skill(entry["model"], [r for i,r in enumerate(LONG_RECORDS) if i in HELD_OUT_RECORDS], CFG.signal, DEVICE,
                            horizons_s=HS_SKILL,
                            n_samples=64 if RUN_MODE == "smoke" else 256,
                            context_windows=CFG.forward.context_windows,
                            seed=CFG.optim.seed, verbose=True)
        entry["model"].cpu()
        if len(df):
            skill_rows.append(df.assign(arm=arm))
    if skill_rows:
        skill_df = pd.concat(skill_rows, ignore_index=True)
        RESULTS["forecast_skill"] = skill_df
        display(skill_df.pivot(index="horizon_s", columns="arm", values="skill").round(4))
        pos = skill_df[skill_df.arm == "dual"]["skill"]
        if len(pos):
            print(f"\nCardio-JEPA (dual) beats persistence at "
                  f"{int((pos > 0).sum())}/{len(pos)} horizons.")
            if (pos <= 0).all():
                print(">>> No horizon shows positive skill: on this run the forward "
                      "predictor is not forecasting beyond copying the context. Train "
                      "longer, or check that long recordings were actually loaded.")
else:
    print("Stage 2c skipped.")
'''

STAGE3_MD = r"""## Stage 3 — RQ3: does the forward objective collapse, and what stops it?

Four arms, identical apart from $\mathcal{L}_{reg}$:

1. **`ema`** — encoder asymmetry + stop-gradient only (what Kim and Weimann & Conrad use).
2. **`vicreg`** — variance hinge + covariance penalty.
3. **`vq`** — soft vector-quantised codebook bottleneck on the targets.
4. **`sigreg`** — LeJEPA/SIGReg; **drops the EMA target network entirely**.

Diagnostics logged every epoch:

* **effective rank (uncentred, RankMe)** — detects *complete* collapse (every sample
  mapping to one vector → rank ≈ 1).
* **covariance effective rank** — detects *dimensional* collapse (directions dying).
  Both are reported because they disagree in an important case: mean-pooling 400
  LayerNormed tokens always leaves a large shared component, so an uncentred rank near 1
  with a healthy covariance rank means "offset-dominated", **not** "dead". Only the
  covariance-rank collapse actually removes what a linear head can use.
* per-dimension variance, mean pairwise cosine similarity;
* ER-JEPA's reported signature — loss falling to ≈0 then partially recovering — is
  detected explicitly rather than eyeballed.

$\beta$ is set per arm (`ANTI_COLLAPSE_DEFAULTS`) so $\mathcal{L}_{reg}$ shapes the
geometry without replacing the objective; an over-strong regulariser trivially prevents
collapse while destroying downstream accuracy, which would be a meaningless "win".
"""

STAGE3_CELL = r'''# --- Stage 3: four-arm anti-collapse study (RQ3) ------------------------------------
STAGE3, stage3_rows = {}, []

if STAGES["stage3_collapse"]:
    for arm in ("ema", "vicreg", "vq", "sigreg"):
        cfg = configure_arm(CFG, arm)
        cfg.loss.weight_forward, cfg.loss.lambda_mask = 1.0, 1.0   # full dual objective
        # RQ3 measures collapse, so here it must be observed rather than aborted on.
        cfg.optim.on_collapse = "warn"

        # The `ema` arm is *identical* to Stage 1's `dual` model -- same objective, same
        # loader, same seed, and L_reg is identically zero for EMAOnly so beta is
        # irrelevant. Reuse it rather than paying for the same run twice.
        reuse = False  # Stage 1 defaults to VICReg; it is not the EMA-only arm
        if reuse:
            print(f"\n=== RQ3 arm [{arm}] -- reusing Stage 1 'dual' (identical setup) ===")
            STAGE3[arm] = {"model": STAGE1["dual"]["model"],
                           "res": STAGE1["dual"]["res"], "cfg": cfg}
        else:
            print(f"\n=== RQ3 arm [{arm}]  beta={cfg.loss.beta_reg}  "
                  f"ema_target={cfg.loss.use_ema_target} ===")
            model = CardioJEPA(cfg.signal, cfg.model, cfg.loss, cfg.hierarchical,
                               cfg.forward)
            maybe_warm_start(model, arm)
            res = cj.pretrain(model, LOADER, cfg, device=DEVICE, arm_name=arm,
                              probe_batch=PROBE_BATCH,
                              save_path=os.path.join(cfg.out_dir, f"rq3_{arm}.pth"),
                              log_every=0 if RUN_MODE == "smoke" else 100)
            STAGE3[arm] = {"model": model, "res": res, "cfg": cfg}

        STAGE3[arm]["model"].cpu()
        s = STAGE3[arm]["res"].monitor.summary()
        row = {"arm": arm, "final_loss": s["final_loss"],
               "erank": s["final_effective_rank"],
               "cov_erank": s["final_cov_effective_rank"],
               "cov_erank_ratio": s["final_cov_erank_ratio"],
               "std_mean": s["final_std_mean"], "cos_sim": s["final_cos_sim"],
               "complete_collapse": s["complete_collapse"],
               "dimensional_collapse": s["dimensional_collapse"],
               "offset_dominated": s["offset_dominated"],
               "loss_to_zero_then_recover": bool(s["collapsed_to_zero"] and s["recovered"])}
        stage3_rows.append(row)

    stage3_df = pd.DataFrame(stage3_rows)
    RESULTS["stage3_collapse"] = stage3_df
    display(stage3_df.round(4))
else:
    print("Stage 3 skipped.")
'''

STAGE3B_CELL = r'''# --- Stage 3b: does anti-collapse buy downstream accuracy? --------------------------
# An arm that prevents collapse but destroys AUROC has not solved anything, so RQ3 is
# only answered by the (stability, accuracy) pair -- the proposal's "Pareto-best" rule.
if STAGE3 and PROBE_SPLITS:
    task_name, splits = next(iter(PROBE_SPLITS.items()))
    rows = []
    for arm, entry in STAGE3.items():
        out = run_classification_probe(entry["model"], splits, DEVICE, CFG.evaluation,
                                       batch_size=64, verbose=False)
        s = entry["res"].monitor.summary()
        rows.append({"arm": arm, "macro_auroc": out["macro_auroc"],
                     "cov_erank_ratio": s["final_cov_erank_ratio"],
                     "std_mean": s["final_std_mean"],
                     "dimensional_collapse": s["dimensional_collapse"]})
        print(f"  {arm:8s} AUROC={out['macro_auroc']:.4f}  "
              f"cov_erank_ratio={s['final_cov_erank_ratio']:.3f}")
    pareto = pd.DataFrame(rows).sort_values("macro_auroc", ascending=False)
    RESULTS["stage3_pareto"] = pareto
    print(f"\nStability vs accuracy on {task_name}:")
    display(pareto.round(4))
    best = pareto.iloc[0]
    print(f"\nHighest observed AUROC (not a Pareto-front calculation): {best['arm']} "
          f"(AUROC {best['macro_auroc']:.4f}, cov_erank_ratio {best['cov_erank_ratio']:.3f})")

    # per-epoch trajectories, for the loss-shape diagnostic
    traj = pd.concat([entry["res"].monitor.to_frame().assign(arm=arm)
                      for arm, entry in STAGE3.items()], ignore_index=True)
    RESULTS["stage3_trajectories"] = traj
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
        for arm, entry in STAGE3.items():
            df = entry["res"].monitor.to_frame()
            axes[0].plot(df.epoch, df.loss, marker="o", label=arm)
            axes[1].plot(df.epoch, df.cov_effective_rank, marker="o", label=arm)
            axes[2].plot(df.epoch, df.std_mean, marker="o", label=arm)
        for ax, t in zip(axes, ("training loss", "covariance effective rank",
                                "mean per-dim std")):
            ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        axes[2].set_yscale("log")
        plt.tight_layout(); plt.show()
    except Exception as exc:
        print("(plot skipped:", exc, ")")
else:
    print("Stage 3b skipped.")
'''

STAGE4_MD = r"""## Stage 4 — H2: hierarchical vs single-scale across horizons

The Level-2 stack pools per-window summary tokens into a minutes-long coarse sequence
and runs a second context/target/predictor triple over it — ER-JEPA's channel→temporal
factorisation, with the temporal *masking* stage replaced by temporal *forecasting*.

H2 makes a directional prediction that is easy to falsify: the hierarchical−single-scale
gap should **widen monotonically as $H$ grows**. A model that only wins at one horizon
has not demonstrated long-horizon modelling. The sweep below is the actual test, and the
monotonicity check at the end reports the verdict either way.
"""

STAGE4_CELL = r'''# --- Stage 4: hierarchical stack (H2) -----------------------------------------------
# Controlled ablation: BOTH models are trained on the same wide-context loader and
# differ only in whether the Level-2 stack is enabled. Reusing Stage 1's model as the
# single-scale arm would confound the comparison, because it was trained with a
# 2-window context and its predictor's window embeddings for slots 2..11 would never
# have received a gradient.
STAGE4, sweep_df, hier_model = {}, None, None

if STAGES["stage4_hierarchical"] and any(r.episodes for r in LONG_RECORDS):
    L2W = CFG.hierarchical.l2_context_windows

    def build_stage4_cfg(name, hierarchical):
        c = copy.deepcopy(CFG)
        c.name = f"{CFG.name}_{name}"
        c.hierarchical.enabled = hierarchical
        c.forward.context_windows = L2W
        return c

    hcfg = build_stage4_cfg("hier", True)
    hier_loader, hier_info = build_pretrain_loaders(
        FOUND, hcfg, long_records=PRETRAIN_LONG_RECORDS,
        samples_per_record=200 if RUN_MODE != "smoke" else 32, verbose=False)
    print(f"wide-context loader: {hier_info['steps_per_epoch']} steps/epoch, "
          f"context = {L2W} x {CFG.signal.window_s}s "
          f"= {L2W * CFG.signal.window_s / 60:.1f} min")

    for name, use_hier in (("hierarchical", True), ("single_scale", False)):
        cfg4 = build_stage4_cfg(name, use_hier)
        print(f"\n=== training [{name}]  level2={use_hier} ===")
        m = CardioJEPA(cfg4.signal, cfg4.model, cfg4.loss, cfg4.hierarchical, cfg4.forward)
        maybe_warm_start(m, name)
        r = cj.pretrain(m, hier_loader, cfg4, device=DEVICE, arm_name=name,
                        probe_batch=PROBE_BATCH,
                        save_path=os.path.join(cfg4.out_dir, f"stage4_{name}.pth"),
                        log_every=0 if RUN_MODE == "smoke" else 100)
        m.cpu()
        STAGE4[name] = {"model": m, "res": r, "cfg": cfg4}
    hier_model = STAGE4["hierarchical"]["model"]

    models = {k: {"model": v["model"], "feature_mode": "world"}
              for k, v in STAGE4.items()}

    HS = ((120.0, 300.0, 600.0) if RUN_MODE == "smoke"
          else (300.0, 900.0, 1800.0, 3600.0))

    def _builder(records, signal, horizon_s, context_windows):
        return build_onset_dataset(
            records, signal, horizon_s=horizon_s, context_windows=context_windows,
            stride_s=CFG.evaluation.anticipation_stride_s,
            guard_s=CFG.evaluation.anticipation_guard_s,
            max_per_record=300 if RUN_MODE != "smoke" else 60)

    print()
    sweep_df = horizon_sweep(models, LONG_RECORDS, CFG.signal, DEVICE, CFG.evaluation,
                             horizons_s=HS, builder=_builder, context_windows=L2W,
                             test_frac=0.3, seed=CFG.optim.seed)
    if len(sweep_df):
        RESULTS["stage4_horizon_sweep"] = sweep_df
        display(sweep_df.round(4))
        piv = sweep_df.pivot(index="horizon_s", columns="model", values="auroc")
        if {"hierarchical", "single_scale"} <= set(piv.columns):
            piv["gap"] = piv["hierarchical"] - piv["single_scale"]
            display(piv.round(4))
            gaps = piv["gap"].dropna().tolist()
            widening = len(gaps) >= 2 and all(b >= a - 1e-6 for a, b in zip(gaps, gaps[1:]))
            print(f"\nH2 -- does the hierarchical gap widen monotonically with horizon? "
                  f"{widening}")
            print("  " + ("consistent with H2" if widening
                          else "NOT consistent with H2 on this run"))
            RESULTS["stage4_h2_monotone"] = bool(widening)
else:
    print("Stage 4 skipped.")
'''

STAGE4B_CELL = r'''# --- Stage 4b: AF-burden regression (the other half of H2) --------------------------
if STAGE4 and any(r.episodes for r in LONG_RECORDS):
    L2W = CFG.hierarchical.l2_context_windows
    bsamples, bstats = build_burden_dataset(
        LONG_RECORDS, CFG.signal, context_windows=L2W, from_first_minutes=30.0,
        samples_per_record=8 if RUN_MODE != "smoke" else 4, seed=CFG.optim.seed)
    print(f"burden dataset: {bstats['n_samples']:.0f} samples over "
          f"{bstats['n_records']:.0f} records, mean AF burden {bstats['mean_burden']:.3f}")

    btr, bte = split_by_record(bsamples, len(LONG_RECORDS), test_frac=0.3,
                               seed=CFG.optim.seed, held_out_records=HELD_OUT_RECORDS)
    rows = []
    for name, entry in STAGE4.items():
        r = run_burden_probe(entry["model"], LONG_RECORDS, btr, bte, CFG.signal, DEVICE,
                             context_windows=L2W, feature_mode="world",
                             horizons_s=[1800.0], batch_size=32, verbose=False)
        rows.append({"model": name, "r2": r.get("r2"), "mae": r.get("mae"),
                     "spearman": r.get("spearman"), "note": r.get("note", "")})
        print(f"  {name:14s} R2={r.get('r2', float('nan')):.3f}  "
              f"MAE={r.get('mae', float('nan')):.3f}  "
              f"rho={r.get('spearman', float('nan')):.3f}")
    if rows:
        burden_df = pd.DataFrame(rows)
        RESULTS["stage4_burden"] = burden_df
        display(burden_df.round(4))
        if len(LONG_RECORDS) < 12:
            print("\n>>> Few records: AF burden is a per-record property, so R2 here is "
                  "estimated from a handful of distinct targets and is very noisy.")
else:
    print("Stage 4b skipped.")
'''

SUMMARY_MD = r"""## Summary and export

Every table is written to `/kaggle/working/results/` (CSV + a JSON manifest) so results
survive the session. Checkpoints and per-epoch diagnostics are under
`/kaggle/working/runs/`.
"""

SUMMARY_CELL = r'''# --- export -------------------------------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
manifest = {"run_mode": RUN_MODE, "config": CFG.to_dict(),
            "datasets_found": {k: v.label for k, v in FOUND.items()},
            "n_long_records": len(LONG_RECORDS),
            "data_info": {k: v for k, v in DATA_INFO.items() if k != "long_records"},
            "held_out_long_record_ids": sorted(HELD_OUT_RECORDS), "tables": [], "scalars": {}}

for name, obj in RESULTS.items():
    if isinstance(obj, pd.DataFrame):
        path = os.path.join(RESULTS_DIR, f"{name}.csv")
        obj.to_csv(path, index=True)
        manifest["tables"].append(f"{name}.csv")
    else:
        manifest["scalars"][name] = obj

with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2, default=str)

print("wrote:")
for f in sorted(os.listdir(RESULTS_DIR)):
    print("  ", os.path.join(RESULTS_DIR, f))

print("\n" + "=" * 78)
print("CARDIO-JEPA RUN SUMMARY")
print("=" * 78)
for name, obj in RESULTS.items():
    print(f"\n--- {name} ---")
    print(obj.round(4).to_string() if isinstance(obj, pd.DataFrame) else obj)
'''

CAVEATS_MD = r"""## Reading these results honestly

* **`smoke` mode proves nothing scientific.** It exists to verify that every code path
  executes. Numbers from it are not evidence about ECG.
* **Synthetic fallback is a pipeline test, not data.** If `discover()` found nothing,
  the model is being trained on Gaussian-mixture beat templates. AUROC there measures
  the generator, not cardiology.
* **The anticipation benchmark here is within-record.** It is Section 6.2 item 3
  (near-term rhythm-change onset), not item 2 (Attia-style 30-day latent-AF), which
  requires longitudinal patient-level diagnosis labels absent from the public mirrors.
  Do not compare these AUROCs to Attia et al.'s 0.87.
* **2-channel Holter data is tiled to 8 leads.** LTAFDB and MIT-BIH are not 12-lead;
  `expand_to_n_leads` replicates channels so one encoder serves both regimes. That is a
  modelling compromise, and cross-lead CroPA attention is degenerate on those records.
* **PTB-XL is read at 100 Hz** (`filename_lr`) by default and upsampled to 250 Hz. Since
  the pipeline bandpasses at 45 Hz nothing in-band is lost, but set `use_hr=True` in
  `index_ptbxl` for the 500 Hz records if you want an exact match to Kim's setup.
* **One seed.** The bootstrap CIs quantify sampling noise in the test set, not training
  variance. Multi-seed runs are needed before any claim of a significant margin.
* **Collapse is arm- and scale-dependent.** Whichever arm wins at `smoke` scale may not
  win at `kaggle` or `full` scale; the RQ3 conclusion should be drawn at the largest
  scale you can afford.

## Mapping back to the proposal

| Proposal | Where it lives |
|---|---|
| §5.2 encoders, CroPA | `models/encoder.py` (`CroPAEncoder`) |
| §5.3(a) $\mathcal{L}_{mask}$ | `models/encoder.py` (`MaskedPredictor`) + `CardioJEPA.forward` |
| §5.3(b) $\mathcal{L}_{fwd}$, horizon token | `models/forward_predictor.py` (`ForwardPredictor`) |
| §5.4 hierarchical Level 2 | `models/cardio_jepa.py` (`Level2Stack`), `SequencePredictor` |
| §5.5 four anti-collapse arms | `regularizers.py`, `ANTI_COLLAPSE_DEFAULTS` |
| §5.5 diagnostics | `diagnostics.py` (`CollapseMonitor`) |
| §5.6 patient-aware sampling | `ForwardSamplingConfig.patient_aware` (hook; needs patient IDs) |
| §6.1 pretraining corpora | `data/discovery.py`, `data/short.py`, `data/longform.py` |
| §6.2 downstream tasks | `eval/benchmarks.py` |
| §6.3 baselines | `models/baselines.py` + the `masked_only` arm |
| §6.4 bootstrap CIs, label efficiency | `eval/stats.py` |
| §7 Stage 1–4 go/no-go | Stages 1–4 above |

### Not implemented here

* **RQ4 / §5.6 cross-modal prediction** (ECG → report-text or PPG latent). None of the
  Kaggle mirrors carry paired reports or synchronous PPG, so there is nothing to predict
  against; adding it means adding a second encoder and a paired corpus.
* **External-site generalisation (§6.2 item 5)** needs a geographically distinct held-out
  cohort with consistent labels — cross-corpus label harmonisation is its own project.
* **Attia-style 30-day latent AF** — see the caveat above.
"""


def build() -> dict:
    cells = [
        md(HEADER), md(DATA_MD),
        md("## 1. Setup"), code(SETUP_CELL),
        md(BOOTSTRAP_MD), code(BOOTSTRAP_CELL),
        md("## 2. Imports and device"), code(IMPORT_CELL),
        md("## 3. Discover attached datasets"), code(DISCOVER_CELL),
        md(CONFIG_MD), code(CONFIG_CELL),
        md(WARMSTART_MD), code(WARMSTART_CELL),
        md("### Build loaders"), code(DATA_BUILD_CELL),
        md(BUDGET_MD), code(BUDGET_CELL),
        md(STAGE1_MD), code(STAGE1_CELL), code(STAGE1_EVAL_CELL),
        md(STAGE2_MD), code(STAGE2_CELL), code(STAGE2B_CELL),
        md(STAGE2C_MD), code(STAGE2C_CELL),
        md(STAGE3_MD), code(STAGE3_CELL), code(STAGE3B_CELL),
        md(STAGE4_MD), code(STAGE4_CELL), code(STAGE4B_CELL),
        md(SUMMARY_MD), code(SUMMARY_CELL),
        md(CAVEATS_MD),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0",
                              "mimetype": "text/x-python",
                              "codemirror_mode": {"name": "ipython", "version": 3},
                              "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                              "file_extension": ".py"},
            "kaggle": {"accelerator": "gpu", "dataSources": [], "isInternetEnabled": False,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join(REPO, "notebooks",
                                                        "cardio_jepa_kaggle.ipynb"))
    args = ap.parse_args()

    nb = build()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")

    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}")
    print(f"  {len(nb['cells'])} cells ({n_code} code), {size_kb:.0f} KB")

    try:
        import nbformat
        nbformat.validate(nbformat.read(args.out, as_version=4))
        print("  nbformat validation: OK")
    except ImportError:
        print("  (nbformat not installed - skipped validation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
