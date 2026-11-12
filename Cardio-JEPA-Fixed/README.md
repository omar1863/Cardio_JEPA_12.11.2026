> **Corrected project edition:** Start with `FIX_REPORT.md` for verified fixes and
> remaining evaluation limitations. For Kaggle import
> `notebooks/Cardio_JEPA_Kaggle_Standalone.ipynb`; attach ECG datasets and enable GPU.
> The standalone notebook contains the corrected package. Original proposal claims
> below should not be treated as independently verified novelty or clinical evidence.

# Cardio-JEPA

Reference implementation of **"Cardio-JEPA: Latent Forward-Prediction Joint-Embedding
Architecture for Cardiac World Modeling from ECG"** — a dual-objective JEPA that adds
*forward-in-time latent prediction* to the masked-infilling paradigm used by every
published ECG-JEPA.

The deliverable is a Kaggle notebook:
**[`notebooks/cardio_jepa_kaggle.ipynb`](notebooks/cardio_jepa_kaggle.ipynb)**.
It imports this package from a **Kaggle Model input** and runs the full experimental
programme — no pip installs, no repo clone, works with internet disabled.

```bash
python tools/export_kaggle_model.py --checkpoint runs/dual.pth
# upload build/kaggle_model/ at https://www.kaggle.com/models -> New Model
#   Framework=PyTorch   Variation=default   Version=1   slug=cardio-jepa
```

The notebook then finds it at
`/kaggle/input/models/<owner>/cardio-jepa/pytorch/default/1`, falling back to
`/kaggle/input/models`, all of `/kaggle/input`, then `/kaggle/working`. Any `.pth`
bundled with it warm-starts the run (`WARM_START = "auto"`).

---

## What is new

Kim's ECG-JEPA (arXiv:2410.08559), Weimann & Conrad's I-JEPA-for-ECG (arXiv:2410.13867)
and the Hierarchical/ER-JEPA (arXiv:2607.01145) all predict masked latent patches
**inside a single static ~10 s recording**. None predicts forward in time. Cardio-JEPA
optimises

```
L  =  w_fwd · L_fwd   +   λ · L_mask   +   β · L_reg
         dynamics          morphology       anti-collapse
          (new)              (Kim)             (RQ3)

L_fwd = ‖ g_φ( f_θ(x[t-L,t]), h ) − sg( f̄_θ(x[t,t+H]) ) ‖
```

where `h` is a learned **horizon token** (2 s … 1 h), so one model is queryable at
multiple forecast horizons.

Because a 10 s clip has no "next window", `L_fwd` runs in two modes that share the same
predictor:

* **cross-window** on long recordings (LTAFDB ~24 h, MIT-BIH 30 min) — genuine
  forecasting at seconds-to-tens-of-minutes horizons;
* **intra-window causal** on 10 s clips — context is the first 60 % of time patches, the
  target is the later patches the encoder never sees. This is the proposal's "short
  intra-recording horizons" de-risking step, and it makes `L_fwd` trainable on all
  ~800k short recordings rather than only on Holter data.

## Code reuse and attribution

| Source | Repository | Reused here |
|---|---|---|
| Kim, ECG-JEPA (2024) | `sehunfromdaegu/ECG_JEPA` | Cross-Pattern Attention encoder, lead-wise masked predictor, 2-D sin-cos positions, EMA schedule, linear-probe protocol |
| Assran et al., I-JEPA (CVPR 2023) | `facebookresearch/ijepa` | ViT blocks, context/target asymmetry, predictor design |
| Na et al., ST-MEM (ICLR 2024) | — | masked raw-reconstruction baseline |

New for this proposal: the forward predictor with horizon tokens, the two-level
hierarchical rhythm stack, the four-arm anti-collapse study, the anticipation benchmarks,
and a dependency-free WFDB reader.

---

## Layout

```
cardio_jepa/
  config.py              experiment configs + smoke/kaggle/full presets
  regularizers.py        RQ3: EMA-only | VICReg | soft-VQ codebook | LeJEPA/SIGReg
  diagnostics.py         effective rank (RankMe + covariance), variance, loss-shape
  engine.py              pretrain loop: cosine LR, EMA schedule, AMP
  models/
    transformer.py       ViT blocks + cross-attention decoder stack
    encoder.py           CroPAEncoder, MaskedPredictor          (from ECG-JEPA)
    forward_predictor.py ForwardPredictor, SequencePredictor    (new)
    cardio_jepa.py       dual objective + Level-2 rhythm stack  (new)
    baselines.py         ST-MEM, xresnet1d, inception1d
  data/
    wfdb_lite.py         dependency-free WFDB reader (212/16/80/32, .mat, .atr)
    discovery.py         auto-detect corpora under /kaggle/input
    short.py             PTB-XL, CPSC2018, Chapman, Ningbo, CODE-15, MIMIC-IV-ECG
    longform.py          LTAFDB / MIT-BIH, forward pairs, anticipation benchmarks
    synthetic.py         synthetic ECG fallback
    build.py             mounts -> loaders and probe splits
  eval/
    linear_probe.py      frozen-encoder probing + label-efficiency sweeps
    stats.py             paired bootstrap CIs on AUROC deltas
    benchmarks.py        classification / anticipation / AF-burden probes
  checkpoint.py          find the package/checkpoints in a Kaggle Model; warm-start
notebooks/cardio_jepa_kaggle.ipynb    generated driver notebook
tools/build_notebook.py               regenerates the notebook
tools/export_kaggle_model.py          packages cardio_jepa/ for upload as a Kaggle Model
tools/check_export.py                 guards notebook/export consistency
tests/test_smoke.py                   end-to-end tests
```

The package is the single source of truth:

```bash
python tools/build_notebook.py        # regenerate the notebook
python tools/export_kaggle_model.py   # regenerate the uploadable package
python tools/check_export.py          # verify the two stay consistent
```

## Local use

```bash
pip install -e .
python -m pytest tests/ -q          # 25 tests, ~35 s on CPU
```

```python
import cardio_jepa as cj
from cardio_jepa.data.build import build_pretrain_loaders, probe_batch_from_loader
from cardio_jepa.models import CardioJEPA

cfg = cj.smoke_config()                       # or kaggle_config() / full_config()
loader, info = build_pretrain_loaders({}, cfg)          # {} -> synthetic fallback
model = CardioJEPA(cfg.signal, cfg.model, cfg.loss, cfg.hierarchical, cfg.forward)
res = cj.pretrain(model, loader, cfg, probe_batch=probe_batch_from_loader(loader))
print(res.monitor.summary())                  # collapse diagnostics
```

## Datasets

Attach via Kaggle's **Add Input → Datasets**. Discovery walks `/kaggle/input` and
matches directory names *at any depth*, so both mount layouts work:

```
/kaggle/input/<slug>/...                            (classic)
/kaggle/input/datasets/<owner>/<slug>/<subdir>/...  (current)
```

Anything missing is skipped, and with nothing attached the notebook falls back to
synthetic ECG so every cell still runs.

| Role (proposal Table 6.1) | Kaggle dataset | Size |
|---|---|---|
| MIMIC-IV-ECG — primary pretraining | `ishtiaqueahmed354/metti-dataset` (**MEETI**) | varies |
| MIMIC-IV-ECG (raw mirrors) | `biswajitkrshaw/mimic-iv-ecg-diagnostic-electrocardiogram` · `beckpro/mimic-iv-ecg-wave-npy-dataset` | 38.5 / 1.7 GB |
| CODE-15 | `gerardolug/ecg-code-15` · `ccf002/code-15-ecg-complete` | 5–46 GB |
| Chapman-Shaoxing + Ningbo | `bjoernjostein/shaoxing-and-ningbo-first-hospital-database` | 2.5 GB |
| CPSC2018 | `physionet/china-physiological-signal-challenge-in-2018` | 0.9 GB |
| Chapman-Shaoxing (alone) | `erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database` | 0.6 GB |
| PTB-XL | `khyeh0719/ptb-xl-dataset` · `omar1863/ptb-xl` | 1.9 GB |
| Long-Term AF (**required** for `L_fwd`) | `sahilshahare12/long-term-af-database` | 1.8 GB |
| MIT-BIH (**required** for `L_fwd`) | `protobioengineering/mit-bih-arrhythmia-database-modern-2023` | 0.2 GB |

A good starting set is PTB-XL + CPSC2018 + Chapman/Ningbo + LTAFDB + MIT-BIH (~7 GB).
**MEETI** is derived from MIMIC-IV-ECG (784,680 12-lead 10 s ECGs from 160,597 ICU
patients) and fills the proposal's primary pretraining slot on its own.

Attaching every mirror at once is not advisable: Kaggle limits attached input and
writable output, and indexing a 38 GB mount costs real minutes before training starts.
Where two mounts resolve to the same directory (a combined Chapman+Ningbo mirror serves
both corpus keys), it is indexed once rather than loading every record twice.

The **MIT-BIH Noise Stress Test** database is *not* the MIT-BIH Arrhythmia Database: it
is 12 deliberately noise-corrupted records plus three pure-noise records (`bw`/`em`/`ma`)
with no AF rhythm annotations. It is classified separately, its noise-only records are
excluded, and it stays out of pretraining unless `cfg.data.use_noise_corpora = True`.

**Without LTAFDB or MIT-BIH Arrhythmia attached**, discovery says so explicitly:
`L_fwd` then trains only via the intra-window path and Stages 2 and 4 fall back to
synthetic recordings. Stages 1 and 3 still produce real results from the short corpora.

## Notebook stages

| Stage | Question | Output |
|---|---|---|
| 1 | Does adding `L_fwd` cost anything on standard classification? | masked-only (Kim reproduction) vs forward-only vs dual vs ST-MEM, PTB-XL/CPSC linear probe + go/no-go bootstrap test |
| 2 (H1) | Does forward pretraining help *anticipation*? | AF-onset probe with leakage control, `context` vs `world` features, label-efficiency curves |
| 3 (RQ3) | Does the forward objective collapse, and what stops it? | four arms × effective-rank/variance/loss-shape diagnostics, plus the stability-vs-accuracy Pareto table |
| 4 (H2) | Does hierarchy help at long horizons? | hierarchical vs single-scale swept across horizons, AF-burden regression |

## Limitations

* The anticipation benchmark is **within-record** near-term onset (proposal §6.2 item 3),
  not Attia-style 30-day latent AF (item 2) — that needs longitudinal patient-level
  diagnosis labels absent from the public mirrors. These AUROCs are not comparable to
  Attia et al.'s 0.87.
* LTAFDB and MIT-BIH are 2-channel; `expand_to_n_leads` tiles them across the 8-lead
  axis so one encoder serves both regimes. Cross-lead CroPA attention is degenerate on
  those records.
* PTB-XL is read at 100 Hz (`filename_lr`) and upsampled to 250 Hz. The pipeline
  bandpasses at 45 Hz so nothing in-band is lost; pass `use_hr=True` to `index_ptbxl`
  for the 500 Hz records.
* Single seed. The bootstrap CIs quantify test-set sampling noise, not training variance.
* RQ4 (cross-modal ECG→report/PPG prediction) and external-site generalisation are not
  implemented — neither has a paired corpus available on Kaggle.
* `smoke` mode and the synthetic fallback exist to verify the pipeline. Numbers from
  either are not evidence about ECG.

## References

Assran et al., *I-JEPA*, CVPR 2023 · Assran et al., *V-JEPA 2*, arXiv:2506.09985 ·
Kim, *ECG-JEPA*, arXiv:2410.08559 · Weimann & Conrad, arXiv:2410.13867 ·
Na et al., *ST-MEM*, ICLR 2024 · Kiyasseh et al., *CLOCS*, ICML 2021 ·
Attia et al., *The Lancet* 394:861–867, 2019 · Garrido et al., *RankMe*, ICML 2023 ·
Balestriero & LeCun, *LeJEPA* (SIGReg).
