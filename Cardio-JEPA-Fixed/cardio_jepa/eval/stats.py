"""Statistical testing for Section 6.4: bootstrap CIs on AUROC differences."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if y_true.ndim == 1:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score.reshape(-1)))
    per = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        per.append(roc_auc_score(y_true[:, j], y_score[:, j]))
    return float(np.mean(per)) if per else float("nan")


def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray, n: int = 2000,
                    alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for a single model's macro AUROC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        v = _auroc(y_true[idx], y_score[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan")}
    vals = np.asarray(vals)
    return {"auroc": _auroc(y_true, y_score),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "std": float(vals.std())}


def bootstrap_auroc_delta(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                          n: int = 2000, alpha: float = 0.05, seed: int = 0
                          ) -> Dict[str, float]:
    """Paired bootstrap on AUROC(a) - AUROC(b).

    Resampling the *same* indices for both models keeps the comparison paired,
    which is what Section 6.4 asks for. ``p_value`` is the two-sided proportion of
    bootstrap deltas on the opposite side of zero (an achieved significance level),
    not a parametric test.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(score_a)
    b = np.asarray(score_b)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        va = _auroc(y_true[idx], a[idx])
        vb = _auroc(y_true[idx], b[idx])
        if not (np.isnan(va) or np.isnan(vb)):
            deltas.append(va - vb)
    if not deltas:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_value": float("nan"), "significant": False}
    d = np.asarray(deltas)
    obs = _auroc(y_true, a) - _auroc(y_true, b)
    lo = float(np.percentile(d, 100 * alpha / 2))
    hi = float(np.percentile(d, 100 * (1 - alpha / 2)))
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"delta": float(obs), "lo": lo, "hi": hi, "p_value": min(p, 1.0),
            "significant": bool(lo > 0 or hi < 0)}


def compare_models(y_true: np.ndarray, scores: Dict[str, np.ndarray],
                   reference: str, n: int = 2000, seed: int = 0):
    """Table of every model vs. ``reference`` with paired-bootstrap CIs."""
    import pandas as pd
    rows = []
    for name, s in scores.items():
        ci = bootstrap_auroc(y_true, s, n=n, seed=seed)
        row = {"model": name, "auroc": ci["auroc"],
               "ci_lo": ci["lo"], "ci_hi": ci["hi"]}
        if name != reference and reference in scores:
            d = bootstrap_auroc_delta(y_true, s, scores[reference], n=n, seed=seed)
            row.update({"delta_vs_ref": d["delta"], "delta_lo": d["lo"],
                        "delta_hi": d["hi"], "p_value": d["p_value"],
                        "significant": d["significant"]})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def label_efficiency_table(results_by_model: Dict[str, Dict[float, float]]):
    """Rows = model, columns = label fraction, values = AUROC (tests H1's
    'margin grows as labels shrink' claim)."""
    import pandas as pd
    df = pd.DataFrame(results_by_model).T
    df.columns = [f"{int(c * 100)}%" for c in df.columns]
    return df
