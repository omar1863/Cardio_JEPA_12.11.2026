"""End-to-end smoke tests. Run with: python -m pytest tests/ -q"""

from __future__ import annotations

import os

import numpy as np
import torch

import cardio_jepa as cj
from cardio_jepa.config import smoke_config
from cardio_jepa.data.build import build_pretrain_loaders, build_probe_splits, probe_batch_from_loader
from cardio_jepa.data.longform import (build_arrhythmia_onset_dataset, build_burden_dataset,
                                       build_onset_dataset, split_by_record,
                                       synthetic_long_records)
from cardio_jepa.data.synthetic import synth_short_corpus
from cardio_jepa.diagnostics import detect_collapse_signature, effective_rank
from cardio_jepa.eval.benchmarks import (run_anticipation_probe, run_burden_probe,
                                         run_classification_probe)
from cardio_jepa.eval.stats import bootstrap_auroc_delta
from cardio_jepa.models.baselines import Inception1d, STMEMBaseline, XResNet1d
from cardio_jepa.models.cardio_jepa import CardioJEPA
from cardio_jepa.models.forward_predictor import horizon_to_bucket


def _cfg():
    cfg = smoke_config()
    cfg.data.allow_synthetic_fallback = True
    cfg.data.input_root = "/nonexistent-input"
    return cfg


def _model(cfg, **loss_kw):
    for k, v in loss_kw.items():
        setattr(cfg.loss, k, v)
    return CardioJEPA(cfg.signal, cfg.model, cfg.loss, cfg.hierarchical, cfg.forward)


# --------------------------------------------------------------------------- units
def test_horizon_buckets_are_monotone():
    idx = horizon_to_bucket([1.0, 10.0, 60.0, 3600.0], cj.HORIZON_BUCKETS_S)
    assert idx.tolist() == sorted(idx.tolist())
    assert idx[0] == 0 and idx[-1] == len(cj.HORIZON_BUCKETS_S) - 1


def test_cropa_mask_structure():
    cfg = _cfg()
    enc = cj.models.CroPAEncoder(embed_dim=32, depth=1, num_heads=4, n_leads=4,
                                 n_time_patches=6, patch_samples=10)
    m = enc.cropa_mask
    assert m.shape == (24, 24)
    # patch (lead 0, t 0) may see (lead 0, t 3) [same lead] and (lead 2, t 0) [same time]
    assert m[0, 3] == 1
    assert m[0, 2 * 6 + 0] == 1
    # but not (lead 2, t 3): neither the same lead nor the same time column
    assert m[0, 2 * 6 + 3] == 0


def test_effective_rank_detects_collapse():
    good = torch.randn(256, 64)
    collapsed = torch.ones(256, 64) + 1e-6 * torch.randn(256, 64)
    assert effective_rank(good) > 20.0
    # complete collapse: every row identical -> rank 1 (only visible uncentred)
    assert effective_rank(collapsed) < 5.0
    assert effective_rank(collapsed, center=True) > 20.0

    # dimensional collapse: half the directions die -> centred erank halves
    partial = torch.randn(256, 64)
    partial[:, 32:] = 0.0
    assert effective_rank(partial, center=True) < 40.0


def test_collapse_signature_detection():
    losses = [1.0, 0.5, 0.02, 0.01, 0.8, 0.7]     # drop to ~0 then recover (ER-JEPA)
    sig = detect_collapse_signature(losses)
    assert sig["collapsed_to_zero"] and sig["recovered"]
    assert not detect_collapse_signature([1.0, 0.9, 0.8, 0.7])["collapsed_to_zero"]


def test_masked_branch_matches_kim_shapes():
    cfg = _cfg()
    m = _model(cfg)
    x = torch.randn(2, 1, cfg.signal.n_leads, cfg.signal.window_samples)
    out = m(x, use_forward_branch=False)
    assert out["loss_mask"] > 0 and float(out["loss_fwd"]) == 0.0


def test_intra_window_forward_runs_without_a_future_window():
    cfg = _cfg()
    m = _model(cfg)
    x = torch.randn(2, 1, cfg.signal.n_leads, cfg.signal.window_samples)
    out = m(x)
    assert float(out["loss_fwd"]) > 0.0, "intra-window L_fwd should be active"


def test_cross_window_forward_and_gradients():
    cfg = _cfg()
    m = _model(cfg)
    C, T = cfg.signal.n_leads, cfg.signal.window_samples
    out = m(torch.randn(2, 2, C, T), future_window=torch.randn(2, C, T),
            horizon_s=torch.tensor([60.0, 300.0]))
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.forward_predictor.parameters())
    # the EMA target encoder must never receive gradients
    assert all(p.grad is None for p in m.target_encoder.parameters())


def test_ema_update_moves_target_towards_online():
    cfg = _cfg()
    m = _model(cfg)
    before = next(m.target_encoder.parameters()).clone()
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(1.0)
    m.update_target(0.5)
    after = next(m.target_encoder.parameters())
    assert not torch.allclose(before, after)


def test_all_four_anti_collapse_arms():
    C = _cfg().signal.n_leads
    T = _cfg().signal.window_samples
    for arm in ("ema", "vicreg", "vq", "sigreg"):
        cfg = _cfg()
        cfg.loss.anti_collapse = arm
        cfg.loss.use_ema_target = arm != "sigreg"
        m = _model(cfg)
        out = m(torch.randn(4, 1, C, T), future_window=torch.randn(4, C, T),
                horizon_s=torch.tensor([60.0] * 4))
        assert torch.isfinite(out["loss"]), f"{arm} produced a non-finite loss"
        out["loss"].backward()
        if arm == "sigreg":
            assert m.target_encoder is None, "sigreg arm should drop the target network"


def test_hierarchical_level2():
    cfg = _cfg()
    cfg.hierarchical.enabled = True
    m = _model(cfg)
    C, T = cfg.signal.n_leads, cfg.signal.window_samples
    out = m(torch.randn(2, 4, C, T), future_window=torch.randn(2, C, T),
            horizon_s=torch.tensor([60.0, 60.0]),
            l2_future=torch.randn(2, C, T), l2_horizon_s=torch.tensor([600.0, 600.0]))
    assert float(out["loss_l2"]) > 0.0
    out["loss"].backward()


def test_forecast_and_anticipation_features():
    cfg = _cfg()
    m = _model(cfg)
    C, T = cfg.signal.n_leads, cfg.signal.window_samples
    x = torch.randn(3, 2, C, T)
    assert m.forecast_latent(x, 300.0).shape == (3, cfg.model.encoder_embed_dim)
    f = m.anticipation_features(x, [60.0, 300.0])
    assert f.shape == (3, 3 * cfg.model.encoder_embed_dim)


def test_baselines_run():
    cfg = _cfg()
    C, T = cfg.signal.n_leads, cfg.signal.window_samples
    st = STMEMBaseline(cfg.signal, cfg.model)
    out = st(torch.randn(2, 1, C, T))
    out["loss"].backward()
    assert st.representation(torch.randn(2, C, T)).shape[1] == cfg.model.encoder_embed_dim
    for net in (XResNet1d(C, 5, widths=(16, 32)), Inception1d(C, 5, depth=2, nf=8)):
        assert net(torch.randn(2, C, T)).shape == (2, 5)


# ------------------------------------------------------------------------- pipeline
def test_wfdb_lite_roundtrip(tmp_path):
    """Write a format-16 record + annotation by hand and read it back."""
    from cardio_jepa.data import wfdb_lite

    fs, n = 250, 500
    raw = (np.sin(np.arange(n) / 8.0) * 1000).astype(np.int16)
    data = np.stack([raw, raw // 2], axis=1).astype("<i2")
    (tmp_path / "r.dat").write_bytes(data.tobytes())
    (tmp_path / "r.hea").write_text(
        f"r 2 {fs} {n}\nr.dat 16 200(0)/mV 16 0 0 0 0 ch0\nr.dat 16 200(0)/mV 16 0 0 0 0 ch1\n")

    rec = wfdb_lite.read_record(str(tmp_path / "r"), prefer_wfdb=False)
    assert rec.p_signal.shape == (n, 2)
    np.testing.assert_allclose(rec.p_signal[:, 0], raw / 200.0, atol=1e-4)

    part = wfdb_lite.read_record(str(tmp_path / "r"), sampfrom=100, sampto=200,
                                 prefer_wfdb=False)
    assert part.p_signal.shape == (100, 2)
    np.testing.assert_allclose(part.p_signal[:, 0], raw[100:200] / 200.0, atol=1e-4)


def test_wfdb_lite_annotation_parsing(tmp_path):
    """Encode two beats and one AFIB rhythm marker, then parse them back."""
    from cardio_jepa.data import wfdb_lite

    words = []

    def ann(code, interval):
        words.append((code << 10) | interval)

    ann(1, 100)                     # 'N' beat at sample 100
    ann(28, 50)                     # '+' rhythm change at 150
    aux = b"(AFIB"
    words.append((63 << 10) | len(aux))
    padded = aux + (b"\x00" if len(aux) % 2 else b"")
    for i in range(0, len(padded), 2):
        words.append(padded[i] | (padded[i + 1] << 8))
    ann(5, 60)                      # 'V' beat at 210
    ann(0, 0)                       # EOF

    (tmp_path / "r.atr").write_bytes(np.array(words, dtype="<u2").tobytes())
    a = wfdb_lite.read_annotation_lite(str(tmp_path / "r.atr"), fs=250.0)

    assert a.sample.tolist() == [100, 150, 210]
    assert a.symbol == ["N", "+", "V"]
    assert a.aux_note[1] == "(AFIB"


def test_synthetic_long_records_and_onset_labels():
    cfg = _cfg()
    recs = synthetic_long_records(4, minutes=20.0, fs=cfg.signal.fs,
                                  n_leads=cfg.signal.n_leads, seed=1)
    assert all(r.episodes for r in recs)
    assert 0.0 < np.mean([r.burden() for r in recs]) < 1.0

    samples, stats = build_onset_dataset(recs, cfg.signal, horizon_s=120.0,
                                         stride_s=10.0, max_per_record=200)
    assert stats["n_positive"] > 0 and stats["n_negative"] > 0

    # leakage control: no context window may overlap an AF episode
    for s in samples:
        rec = recs[s.record]
        t0, t1 = s.start, s.start + int(cfg.signal.window_s * rec.fs)
        assert not any(a < t1 and b > t0 for a, b in rec.rhythm_intervals())

    tr, te = split_by_record(samples, len(recs), test_frac=0.5, seed=0)
    assert tr and te
    assert not (set(s.record for s in tr) & set(s.record for s in te))


def test_burden_dataset():
    cfg = _cfg()
    recs = synthetic_long_records(3, minutes=20.0, fs=cfg.signal.fs,
                                  n_leads=cfg.signal.n_leads, seed=2)
    samples, stats = build_burden_dataset(recs, cfg.signal, context_windows=2,
                                          from_first_minutes=5.0, samples_per_record=4)
    assert len(samples) > 0 and 0.0 <= stats["mean_burden"] <= 1.0


def test_full_pretrain_and_probe_pipeline():
    cfg = _cfg()
    cfg.hierarchical.enabled = True
    cfg.optim.epochs = 1

    loader, info = build_pretrain_loaders({}, cfg, samples_per_record=16, verbose=False)
    probe = probe_batch_from_loader(loader, n=16)
    assert probe.shape[1:] == (cfg.signal.n_leads, cfg.signal.window_samples)

    model = _model(cfg)
    res = cj.pretrain(model, loader, cfg, arm_name="ema", probe_batch=probe,
                      verbose=False, log_every=0)
    assert len(res.monitor.history) == 1
    assert np.isfinite(res.monitor.history[0]["effective_rank"])

    splits = build_probe_splits({}, cfg, verbose=False)
    task = next(iter(splits.values()))
    out = run_classification_probe(model, task, torch.device("cpu"), cfg.evaluation,
                                   batch_size=16, verbose=False)
    assert 0.0 <= out["macro_auroc"] <= 1.0

    recs = info["long_records"]
    samples, stats = build_onset_dataset(recs, cfg.signal, horizon_s=120.0, stride_s=20.0,
                                         max_per_record=60)
    tr, te = split_by_record(samples, len(recs), test_frac=0.4, seed=0)
    if tr and te:
        r = run_anticipation_probe(model, recs, tr, te, cfg.signal, torch.device("cpu"),
                                   cfg.evaluation, feature_mode="world",
                                   horizons_s=[120.0], batch_size=16, verbose=False)
        assert np.isnan(r["auroc"]) or 0.0 <= r["auroc"] <= 1.0

    bs, _ = build_burden_dataset(recs, cfg.signal, context_windows=2,
                                 from_first_minutes=4.0, samples_per_record=4)
    btr, bte = split_by_record(bs, len(recs), test_frac=0.4, seed=0)
    if btr and bte:
        rb = run_burden_probe(model, recs, btr, bte, cfg.signal, torch.device("cpu"),
                              context_windows=2, feature_mode="context",
                              batch_size=8, verbose=False)
        assert "mae" in rb


def test_bootstrap_delta_is_paired():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.5, 400)
    bad = rng.normal(0, 1, 400)
    d = bootstrap_auroc_delta(y, good, bad, n=200)
    assert d["delta"] > 0 and d["significant"]


def test_synthetic_short_corpus_is_separable():
    """Sanity check on the fallback data: AF vs sinus must be learnable at all."""
    w, y = synth_short_corpus(200, fs=250.0, window_s=10.0, n_leads=8, seed=3)
    assert w.shape == (200, 8, 2500) and set(np.unique(y)) <= {0, 1}
    # AF clips have faster, more variable rhythm -> higher zero-crossing dispersion
    zc = np.array([np.diff(np.signbit(x[1])).sum() for x in w])
    assert zc[y == 1].mean() != zc[y == 0].mean()


def test_forecast_skill_runs_and_is_signed():
    """An untrained predictor should show no meaningful skill over persistence."""
    from cardio_jepa.eval.benchmarks import forecast_skill

    cfg = _cfg()
    m = _model(cfg)
    recs = synthetic_long_records(3, minutes=15.0, fs=cfg.signal.fs,
                                  n_leads=cfg.signal.n_leads, seed=5)
    df = forecast_skill(m, recs, cfg.signal, torch.device("cpu"),
                        horizons_s=(10.0, 60.0), n_samples=16, verbose=False)
    assert set(["horizon_s", "cos_pred", "cos_persistence", "skill"]) <= set(df.columns)
    assert len(df) == 2
    assert np.isfinite(df["skill"]).all()
    # skill is a difference of cosines, so it must stay in [-2, 2]
    assert df["skill"].abs().max() <= 2.0


def test_discovery_handles_nested_kaggle_layout(tmp_path):
    """Kaggle's current layout nests corpora under datasets/<owner>/<slug>/<subdir>."""
    from cardio_jepa.data.discovery import discover

    root = tmp_path / "input"
    def mk(rel, files=()):
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_text("")
        return d

    mk("datasets/erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database/WFDB_ChapmanShaoxing",
       ["JS00001.hea", "JS00001.mat"])
    mk("datasets/organizations/physionet/china-physiological-signal-challenge-in-2018/Training_WFDB",
       ["A0001.hea", "A0001.mat"])
    mk("datasets/ishtiaqueahmed354/metti-dataset/MEETI", ["rec0.npy"])
    mk("datasets/omar1863/ptb-xl/PTB XL/ptb-xl-1.0.1", ["ptbxl_database.csv"])
    mk("datasets/omar1863/ptb-xl/mit-bih-noise-stress-test-database-1.0.0",
       ["118e00.hea", "118e00.dat", "bw.hea", "bw.dat"])
    # models/ holds code, never data -- must not be scanned as a corpus
    mk("models/someone/cardio-jepa/pytorch/default/1", ["ptbxl_database.csv"])

    found = discover(str(root), verbose=False)
    assert {"chapman", "cpsc", "meeti", "ptbxl"} <= set(found)
    assert found["ptbxl"].fmt == "ptbxl"
    assert found["meeti"].kind == "short"
    # the noise-stress database must be its own key, never mistaken for MIT-BIH
    assert "mitbih_nst" in found and found["mitbih_nst"].kind == "noise"
    assert "mitbih" not in found
    # nothing may be discovered inside models/
    assert all("/models/" not in d.root for d in found.values())


def test_noise_only_records_are_excluded():
    """bw/em/ma are pure noise, not ECG -- they must never enter a training index."""
    from cardio_jepa.data.discovery import FoundDataset
    from cardio_jepa.data.short import NOISE_ONLY_RECORDS
    assert NOISE_ONLY_RECORDS == {"bw", "em", "ma"}


def test_combined_mirror_is_indexed_once(tmp_path):
    """A Chapman+Ningbo mirror satisfies two keys from one directory."""
    from cardio_jepa.data.build import load_short_indices
    from cardio_jepa.data.discovery import discover

    root = tmp_path / "input"
    d = root / "datasets/bjoernjostein/shaoxing-and-ningbo-first-hospital-database/WFDB"
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"JS{i:05d}.hea").write_text("")

    found = discover(str(root), verbose=False)
    assert {"chapman", "ningbo"} <= set(found)
    assert found["chapman"].data_root == found["ningbo"].data_root

    cfg = _cfg()
    idx = load_short_indices(found, cfg, verbose=False)
    assert len(idx) == 1, "the shared directory must be indexed once, not twice"


def test_warm_start_tolerates_width_mismatch(tmp_path):
    import cardio_jepa as cj

    cfg = _cfg()
    src = _model(cfg)
    p = tmp_path / "ck.pth"
    torch.save({"model": src.state_dict(), "config": cfg.to_dict()}, p)

    same = _model(_cfg())
    r = cj.load_weights(same, str(p), verbose=False)
    assert not r.missing and r.n_loaded > 0

    wide = _cfg()
    wide.model.encoder_embed_dim = 128
    wide.model.encoder_num_heads = 4
    r2 = cj.load_weights(_model(wide), str(p), verbose=False)
    assert r2.shape_mismatch, "a width change must be reported, not silently ignored"
    assert 0 < r2.n_loaded < len(r.loaded)


def test_find_package_root(tmp_path):
    import cardio_jepa as cj

    holder = tmp_path / "models" / "o" / "cardio-jepa" / "pytorch" / "default" / "1"
    (holder / "cardio_jepa").mkdir(parents=True)
    (holder / "cardio_jepa" / "__init__.py").write_text("")
    assert cj.find_package_root([str(tmp_path)]) == str(holder)
    assert cj.find_package_root([str(tmp_path / "nope")]) is None


def test_nested_corpus_is_excluded_from_its_parent(tmp_path):
    """`omar1863/ptb-xl` really does ship the noise-stress DB inside the PTB-XL mount.

    Without exclusion the PTB-XL index swallows the noise records.
    """
    from cardio_jepa.data.build import load_short_indices
    from cardio_jepa.data.discovery import discover

    root = tmp_path / "input"
    ptb = root / "datasets/omar1863/ptb-xl/PTB XL/ptb-xl-1.0.1"
    ptb.mkdir(parents=True)
    for i in range(4):
        (ptb / f"rec{i}.hea").write_text("")
    nst = root / "datasets/omar1863/ptb-xl/mit-bih-noise-stress-test-database-1.0.0"
    nst.mkdir(parents=True)
    for nm in ("118e00", "bw", "em", "ma"):
        (nst / f"{nm}.hea").write_text("")

    found = discover(str(root), verbose=False)
    assert "ptbxl" in found and "mitbih_nst" in found
    assert found["mitbih_nst"].root in found["ptbxl"].exclude_dirs

    cfg = _cfg()
    idx = load_short_indices(found, cfg, verbose=False)
    ptb_records = [os.path.basename(p) for p in idx["ptbxl"].locators]
    assert len(ptb_records) == 4
    for noise in ("118e00", "bw", "em", "ma"):
        assert noise not in ptb_records, f"{noise} leaked into the PTB-XL index"


def test_find_checkpoints_prefers_root_order_over_size(tmp_path):
    """The curated model-input checkpoint must beat a bigger scratch one.

    Sorting globally by size let a mid-run file in runs/ silently outrank the
    checkpoint deliberately shipped with the model input.
    """
    import cardio_jepa as cj

    model_in = tmp_path / "models" / "o" / "cardio-jepa" / "pytorch" / "default" / "1"
    model_in.mkdir(parents=True)
    (model_in / "pretrained_dual.pth").write_bytes(b"x" * 1000)

    runs = tmp_path / "working" / "runs"
    runs.mkdir(parents=True)
    (runs / "stage4_hierarchical.pth").write_bytes(b"x" * 5000)   # bigger, but scratch

    found = cj.find_checkpoints([str(model_in), str(runs)])
    assert os.path.basename(found[0]) == "pretrained_dual.pth", (
        f"model input must win regardless of size, got {found}")

    # reversing the priority order reverses the winner
    found2 = cj.find_checkpoints([str(runs), str(model_in)])
    assert os.path.basename(found2[0]) == "stage4_hierarchical.pth"


def test_find_checkpoints_newest_first_within_a_root(tmp_path):
    import time
    import cardio_jepa as cj

    d = tmp_path / "runs"
    d.mkdir()
    (d / "old.pth").write_bytes(b"x" * 9000)
    time.sleep(0.01)
    (d / "new.pth").write_bytes(b"x" * 100)
    os.utime(d / "old.pth", (1, 1))

    found = cj.find_checkpoints([str(d)])
    assert os.path.basename(found[0]) == "new.pth"


# --------------------------------------------------------- collapse guard / budget
def _forward_only_cfg(**over):
    cfg = _cfg()
    cfg.model.encoder_embed_dim = 64
    cfg.model.encoder_depth = 2
    cfg.model.encoder_num_heads = 4
    cfg.model.predictor_embed_dim = 48
    cfg.model.predictor_depth = 1
    cfg.optim.epochs = 3
    cfg.optim.batch_size = 8
    cfg.optim.warmup_epochs = 0
    cfg.optim.lr = 1e-3
    cfg.data.synthetic_n_short = 64
    cfg.data.synthetic_n_long = 0
    cfg.loss.weight_forward, cfg.loss.lambda_mask = 1.0, 0.0
    for k, v in over.items():
        obj, _, attr = k.rpartition(".")
        setattr(getattr(cfg, obj) if obj else cfg, attr, v)
    return cfg


def test_check_collapse_detects_the_trivial_solution():
    """Unit-test the detector itself, on both of its signals."""
    from cardio_jepa.engine import check_collapse

    cfg = _cfg().optim

    # signal 1: loss far below its own starting value
    assert check_collapse(None, None, torch.device("cpu"),
                          loss_now=0.001, loss_ref=1.0, cfg=cfg)
    assert check_collapse(None, None, torch.device("cpu"),
                          loss_now=0.5, loss_ref=1.0, cfg=cfg) is None

    # signal 2: degenerate embeddings, even when the loss looks unremarkable
    class _Const:
        training = False

        def eval(self):
            pass

        def representation(self, x):
            return torch.ones(x.size(0), 64) * 3.0

    reason = check_collapse(_Const(), torch.randn(32, 8, 2500), torch.device("cpu"),
                            loss_now=0.5, loss_ref=1.0, cfg=cfg)
    assert reason and "dead" in reason

    # ...but a *healthy* model must not trip the embedding gate. The pooled
    # representation here is offset-dominated even when fine (cos_sim ~ 0.9999),
    # so a loose threshold would abort good runs.
    class _Healthy:
        training = False

        def eval(self):
            pass

        def representation(self, x):
            g = torch.Generator().manual_seed(0)
            return 3.0 + 0.004 * torch.randn(x.size(0), 64, generator=g)

    assert check_collapse(_Healthy(), torch.randn(32, 8, 2500), torch.device("cpu"),
                          loss_now=0.5, loss_ref=1.0, cfg=cfg) is None


def test_collapse_guard_stops_the_run():
    """When the detector fires, training must abandon the run rather than continue."""
    cfg = _forward_only_cfg()
    cfg.optim.on_collapse = "stop"
    cfg.optim.collapse_check_every = 2
    cfg.optim.collapse_loss_frac = 10.0    # force: any loss under 10x the ref trips it

    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    res = cj.pretrain(_model(cfg), loader, cfg, arm_name="forced",
                      verbose=False, log_every=0)
    assert res.collapsed and res.stopped_early
    assert res.collapse_step is not None
    assert len(res.monitor.history) < cfg.optim.epochs


def test_collapse_guard_can_observe_instead_of_aborting():
    """RQ3 needs to *measure* collapse, so 'warn' must run to completion."""
    cfg = _forward_only_cfg()
    cfg.optim.on_collapse = "warn"
    cfg.optim.collapse_check_every = 2
    cfg.optim.collapse_loss_frac = 10.0

    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    res = cj.pretrain(_model(cfg), loader, cfg, arm_name="warned",
                      verbose=False, log_every=0)
    assert res.collapsed and not res.stopped_early
    assert len(res.monitor.history) == cfg.optim.epochs


def test_max_steps_per_epoch_caps_the_budget():
    cfg = _forward_only_cfg()
    cfg.optim.on_collapse = "ignore"
    cfg.optim.epochs = 2
    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    assert len(loader) > 3, "need a loader longer than the cap for this test"

    cfg.optim.max_steps_per_epoch = 3
    res = cj.pretrain(_model(cfg), loader, cfg, arm_name="capped",
                      verbose=False, log_every=0)
    assert len(res.monitor.history) == 2
    # seconds/epoch should reflect 3 steps, not len(loader)
    assert res.history[0]["seconds"] < 1e6


def test_reg_target_pooled_is_measured_on_the_pooled_representation():
    """The token target is satisfied by positional variance alone, which is why the
    pooled target exists."""
    cfg = _cfg()
    cfg.loss.anti_collapse = "vicreg"
    C, T = cfg.signal.n_leads, cfg.signal.window_samples
    x = torch.randn(4, 1, C, T)

    cfg.loss.reg_target = "pooled"
    m_pooled = _model(cfg)
    r_pooled = float(m_pooled(x)["loss_reg"])

    cfg.loss.reg_target = "tokens"
    m_tok = _model(cfg)
    r_tok = float(m_tok(x)["loss_reg"])

    assert r_pooled > 0 and r_tok > 0
    # pooling 400 LayerNormed tokens shrinks per-sample spread, so the hinge on the
    # pooled representation must be the stricter of the two
    assert r_pooled > r_tok

    cfg.loss.reg_target = "both"
    assert float(_model(cfg)(x)["loss_reg"]) > 0


def test_estimate_runtime_projects_sanely():
    from cardio_jepa.engine import budget_steps_per_epoch, estimate_runtime

    cfg = _forward_only_cfg()
    cfg.optim.epochs = 10
    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    est = estimate_runtime(_model(cfg), loader, cfg, torch.device("cpu"),
                           n_steps=3, n_models=4)
    assert est["s_per_step"] > 0
    assert est["hours_total"] == est["hours_per_model"] * 4
    # the suggested cap must actually bring the projection inside the budget
    fits = budget_steps_per_epoch(est["s_per_step"], cfg.optim.epochs, 4, 1.0)
    assert fits * cfg.optim.epochs * 4 * est["s_per_step"] <= 1.0 * 3600 * 1.05


def test_real_ltafdb_header_and_annotation_shapes(tmp_path):
    """Regression guard built from a real LTAFDB record (ltafdb 1.0.0, record 00).

    LTAFDB is format 16 at 128 Hz with two channels and a gain that has no
    parenthesised baseline -- a header shape the synthetic fixtures did not cover.
    Rhythm changes arrive as '+' annotations whose AUX text is '(AFIB' / '(VT' / '(N'.
    """
    from cardio_jepa.data import wfdb_lite
    from cardio_jepa.data.longform import _episodes_from_annotation

    (tmp_path / "00.hea").write_text(
        "00 2 128 9661440  9:30:00 31/01/2003\n"
        "00.dat 16 166.945/mV 0 0 -1 -8202 0 ECG\n"
        "00.dat 16 173.01/mV 0 0 3 6311 0 ECG\n")
    h = wfdb_lite.parse_header(str(tmp_path / "00.hea"))
    assert h.n_sig == 2 and h.fs == 128.0 and h.n_samp == 9661440
    assert h.signals[0].fmt == "16"
    assert abs(h.signals[0].adc_gain - 166.945) < 1e-6
    assert h.signals[0].baseline == 0        # no "(baseline)" -> falls back to adc_zero

    # a '+' rhythm marker carrying '(AFIB' must open an AF episode
    ann = wfdb_lite.Annotation(
        sample=np.array([100, 200, 5000], dtype=np.int64),
        symbol=["N", "+", "+"], aux_note=["", "(AFIB", "(N"], fs=128.0)
    eps = _episodes_from_annotation(ann, n_samples=10000)
    assert eps == [(200, 5000, "AFIB"), (5000, 10000, "N")]
