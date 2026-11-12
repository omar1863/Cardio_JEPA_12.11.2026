"""Linear probing on a frozen encoder (the primary comparison, Section 6.4).

Linear probing removes fine-tuning confounds, so every method is compared on the
representation it actually learned. Adapted from ECG-JEPA's ``linear_probe_utils.py``
(precompute features once, then train a cheap linear head).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def precompute_features(encode: Callable[[torch.Tensor], torch.Tensor],
                        loader: DataLoader, device, verbose: bool = False
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for i, batch in enumerate(loader):
        x, y = batch if isinstance(batch, (tuple, list)) else (batch, None)
        x = x.to(device)
        f = encode(x)
        feats.append(f.detach().float().cpu())
        if y is not None:
            labels.append(y.detach().cpu())
        if verbose and i % 20 == 0:
            print(f"    features {i * loader.batch_size}...")
    F = torch.cat(feats) if feats else torch.zeros(0)
    Y = torch.cat(labels) if labels else torch.zeros(0)
    return F, Y


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, batch_norm: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim, affine=False, eps=1e-6) if batch_norm else nn.Identity()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(self.bn(x))


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro AUROC over label columns, skipping single-class columns."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 1:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score.reshape(-1)))
    scores = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, j], y_score[:, j]))
    return float(np.mean(scores)) if scores else float("nan")


def train_linear_probe(train_feats: torch.Tensor, train_labels: torch.Tensor,
                       test_feats: torch.Tensor, test_labels: torch.Tensor,
                       n_classes: int, epochs: int = 40, lr: float = 1e-3,
                       weight_decay: float = 1e-4, batch_size: int = 128,
                       device=None, multilabel: bool = True,
                       label_fraction: float = 1.0, seed: int = 0,
                       verbose: bool = False) -> Dict[str, object]:
    """Fit a linear head on frozen features; returns metrics and test scores."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(train_feats) == 0 or len(test_feats) == 0:
        raise ValueError("Probe requires nonempty training and test features")
    if not 0 < label_fraction <= 1:
        raise ValueError("label_fraction must be in (0, 1]")
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    if label_fraction < 1.0:
        n = max(n_classes * 2, int(round(len(train_feats) * label_fraction)))
        n = min(n, len(train_feats))
        idx = torch.randperm(len(train_feats), generator=g)[:n]
        train_feats, train_labels = train_feats[idx], train_labels[idx]

    ds = TensorDataset(train_feats, train_labels.float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=len(ds) > batch_size)

    head = LinearHead(train_feats.shape[1], n_classes, batch_norm=len(ds) > 1 and batch_size > 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

    for _ in range(epochs):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = head(xb)
            loss = crit(out, yb if multilabel else yb.long().squeeze(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    head.eval()
    with torch.no_grad():
        logits = head(test_feats.to(device)).cpu().numpy()
    y_true = test_labels.numpy()

    if multilabel:
        scores = 1.0 / (1.0 + np.exp(-logits))
        auroc = macro_auroc(y_true, logits)
    else:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores = e / e.sum(axis=1, keepdims=True)
        auroc = macro_auroc(np.eye(n_classes)[y_true.astype(int).reshape(-1)], scores)

    out: Dict[str, object] = {"macro_auroc": auroc, "y_true": y_true,
                              "y_score": scores, "n_train": len(train_feats),
                              "label_fraction": label_fraction}
    if multilabel and y_true.ndim == 2:
        from sklearn.metrics import f1_score
        out["macro_f1"] = float(f1_score(y_true, (scores >= 0.5).astype(int),
                                         average="macro", zero_division=0))
    if verbose:
        print(f"    linear probe: macro AUROC = {auroc:.4f} "
              f"(n_train={len(train_feats)}, frac={label_fraction})")
    return out


def probe_encoder(encode: Callable[[torch.Tensor], torch.Tensor],
                  train_ds, test_ds, n_classes: int, device,
                  batch_size: int = 64, epochs: int = 40, lr: float = 1e-3,
                  multilabel: bool = True,
                  label_fractions: Sequence[float] = (1.0,),
                  num_workers: int = 0, seed: int = 0,
                  verbose: bool = True) -> Dict[str, object]:
    """Precompute features once, then sweep label fractions (label-efficiency curve)."""
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    te_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    if verbose:
        print(f"    extracting features: {len(train_ds)} train / {len(test_ds)} test")
    Ftr, Ytr = precompute_features(encode, tr_loader, device)
    Fte, Yte = precompute_features(encode, te_loader, device)

    results: Dict[str, object] = {"feature_dim": int(Ftr.shape[1]) if len(Ftr) else 0,
                                 "by_fraction": {}}
    for frac in label_fractions:
        r = train_linear_probe(Ftr, Ytr, Fte, Yte, n_classes=n_classes, epochs=epochs,
                               lr=lr, device=device, multilabel=multilabel,
                               label_fraction=float(frac), seed=seed, verbose=verbose)
        results["by_fraction"][float(frac)] = r
    full = results["by_fraction"].get(1.0) or list(results["by_fraction"].values())[-1]
    results["macro_auroc"] = full["macro_auroc"]
    results["y_true"] = full["y_true"]
    results["y_score"] = full["y_score"]
    return results
