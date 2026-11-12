"""Loading the package and pretrained weights from a Kaggle Model input.

Kaggle mounts a Model at::

    /kaggle/input/models/<owner>/<model>/<framework>/<variation>/<version>/

which can hold the ``cardio_jepa`` package source, saved checkpoints, or both.
This module covers the two things a notebook needs:

  * ``find_package_root``   -- locate an importable ``cardio_jepa`` package so the
    notebook can ``sys.path`` to it instead of writing the source out itself.
  * ``load_weights``        -- warm-start a model from a previous run's checkpoint,
    tolerating shape and key mismatches so a run at a different width still loads
    whatever genuinely matches.

The bootstrap half of ``find_package_root`` is duplicated inline in the notebook's
first cell -- it has to be, because it runs *before* the package is importable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_MODEL_ROOTS = ("/kaggle/input/models", "/kaggle/input")
CHECKPOINT_EXTS = (".pth", ".pt", ".ckpt", ".bin", ".safetensors")


def find_package_root(search_roots: Sequence[str] = DEFAULT_MODEL_ROOTS,
                      max_depth: int = 8) -> Optional[str]:
    """Return a directory D such that ``D/cardio_jepa/__init__.py`` exists."""
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        stack: List[Tuple[str, int]] = [(root, 0)]
        while stack:
            path, depth = stack.pop()
            if depth > max_depth:
                continue
            cand = os.path.join(path, "cardio_jepa", "__init__.py")
            if os.path.isfile(cand):
                return path
            try:
                for e in os.scandir(path):
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                        stack.append((e.path, depth + 1))
            except (PermissionError, FileNotFoundError, NotADirectoryError):
                continue
    return None


def find_checkpoints(search_roots: Sequence[str] = DEFAULT_MODEL_ROOTS,
                     max_depth: int = 8) -> List[str]:
    """Checkpoint-looking files under the given roots, **in root order**.

    Root order is the priority: callers pass the curated model input first and
    scratch directories like ``runs/`` last, so a checkpoint deliberately shipped
    with the model always outranks whatever an earlier session happened to leave
    lying around. Sorting globally by size instead -- as this used to -- let a
    mid-run scratch checkpoint that is a few KB larger silently win, which is the
    opposite of what "auto" should mean.

    Within one root, newest first (most recent training run), then largest.
    """
    paths: List[str] = []
    seen: set = set()

    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        here: List[Tuple[float, int, str]] = []
        stack: List[Tuple[str, int]] = [(root, 0)]
        while stack:
            path, depth = stack.pop()
            if depth > max_depth:
                continue
            try:
                entries = list(os.scandir(path))
            except (PermissionError, FileNotFoundError, NotADirectoryError):
                continue
            for e in entries:
                if e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                    stack.append((e.path, depth + 1))
                elif e.is_file() and e.name.lower().endswith(CHECKPOINT_EXTS):
                    try:
                        st = e.stat()
                        here.append((st.st_mtime, st.st_size, e.path))
                    except OSError:
                        here.append((0.0, 0, e.path))
        here.sort(reverse=True)
        for _mtime, _size, p in here:
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                paths.append(p)
    return paths


def _extract_state_dict(obj) -> Dict[str, object]:
    """Unwrap the several shapes a torch checkpoint can arrive in."""
    if not isinstance(obj, dict):
        return {}
    for key in ("model", "state_dict", "encoder", "module"):
        inner = obj.get(key)
        if isinstance(inner, dict) and inner:
            # a nested wrapper, or already the tensors themselves
            if all(hasattr(v, "shape") for v in list(inner.values())[:4]):
                return inner
            return _extract_state_dict(inner)
    if all(hasattr(v, "shape") for v in list(obj.values())[:4] or [None]):
        return obj
    return {}


@dataclass
class LoadReport:
    path: str = ""
    loaded: List[str] = field(default_factory=list)
    shape_mismatch: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    checkpoint_config: Optional[dict] = None

    @property
    def n_loaded(self) -> int:
        return len(self.loaded)

    def summary(self) -> str:
        total = len(self.loaded) + len(self.missing)
        pct = 100.0 * len(self.loaded) / max(total, 1)
        lines = [f"loaded {len(self.loaded)}/{total} tensors ({pct:.1f}%) from "
                 f"{os.path.basename(self.path)}"]
        if self.shape_mismatch:
            lines.append(f"  {len(self.shape_mismatch)} skipped on shape mismatch "
                         f"(e.g. {self.shape_mismatch[0]})")
        if self.missing:
            lines.append(f"  {len(self.missing)} left at initialisation "
                         f"(e.g. {self.missing[0]})")
        if self.unexpected:
            lines.append(f"  {len(self.unexpected)} in checkpoint but not in model "
                         f"(e.g. {self.unexpected[0]})")
        if pct < 50.0:
            lines.append("  WARNING: under half the model was restored - the "
                         "checkpoint's architecture probably differs from this config, "
                         "so this is closer to a fresh start than a warm one.")
        return "\n".join(lines)


def load_weights(model, path: str, strict: bool = False,
                 map_location: str = "cpu", verbose: bool = True) -> LoadReport:
    """Warm-start ``model`` from a checkpoint, skipping anything that does not fit.

    Returns a report rather than raising, because the common real case -- resuming a
    run at a different encoder width -- legitimately restores only part of the model,
    and silently loading 3% of the tensors while calling it a warm start would be
    worse than saying so.
    """
    import torch

    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                                   # torch < 2.0 has no weights_only
        obj = torch.load(path, map_location=map_location)

    report = LoadReport(path=path)
    if isinstance(obj, dict):
        report.checkpoint_config = obj.get("config") if isinstance(
            obj.get("config"), dict) else None

    src = _extract_state_dict(obj)
    if not src:
        if verbose:
            print(f"[checkpoint] {path}: no tensors found inside")
        return report

    # tolerate a `module.` prefix from DataParallel-saved checkpoints
    if all(k.startswith("module.") for k in src):
        src = {k[len("module."):]: v for k, v in src.items()}

    dst = model.state_dict()
    to_load: Dict[str, object] = {}
    for k, v in dst.items():
        if k not in src:
            report.missing.append(k)
            continue
        sv = src[k]
        if getattr(sv, "shape", None) == getattr(v, "shape", None):
            to_load[k] = sv
            report.loaded.append(k)
        else:
            report.shape_mismatch.append(f"{k}: {tuple(sv.shape)} vs {tuple(v.shape)}")
            report.missing.append(k)
    report.unexpected = [k for k in src if k not in dst]

    if strict and (report.missing or report.unexpected):
        raise RuntimeError(
            f"strict load failed: {len(report.missing)} missing, "
            f"{len(report.unexpected)} unexpected")

    dst.update(to_load)
    model.load_state_dict(dst)
    if verbose:
        print("[checkpoint] " + report.summary())
    return report


def describe_model_input(root: str) -> str:
    """Human-readable inventory of a Kaggle Model directory."""
    if not os.path.isdir(root):
        return f"{root}: not mounted"
    lines = [f"{root}:"]
    n_py = 0
    pkg = None
    ckpts: List[Tuple[str, float]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                       and d != "__pycache__"]
        if os.path.basename(dirpath) == "cardio_jepa" and "__init__.py" in filenames:
            pkg = dirpath
        for fn in filenames:
            if fn.endswith(".py"):
                n_py += 1
            elif fn.lower().endswith(CHECKPOINT_EXTS):
                full = os.path.join(dirpath, fn)
                try:
                    ckpts.append((full, os.path.getsize(full) / 1e6))
                except OSError:
                    ckpts.append((full, 0.0))
    lines.append(f"  cardio_jepa package : {pkg or 'not found'}")
    lines.append(f"  python files        : {n_py}")
    lines.append(f"  checkpoints         : {len(ckpts)}")
    for p, mb in sorted(ckpts, key=lambda x: -x[1])[:8]:
        lines.append(f"      {mb:8.1f} MB  {os.path.relpath(p, root)}")
    return "\n".join(lines)
