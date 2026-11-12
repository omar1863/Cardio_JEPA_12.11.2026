"""Auto-discovery of ECG corpora mounted under ``/kaggle/input``.

Kaggle does not use one fixed mount layout. Depending on how an input was added
you get either::

    /kaggle/input/<slug>/...                                    (classic)
    /kaggle/input/datasets/<owner>/<slug>/<subdir>/...          (current)
    /kaggle/input/models/<owner>/<model>/<fw>/<var>/<ver>/...   (models)

so we cannot just keyword-match the top-level directory names -- doing that on the
current layout matches only ``datasets`` and ``models`` and finds nothing. Instead
we walk directories (not files) to a bounded depth and match each directory *name*
against a corpus pattern, keeping the shallowest match per corpus.

Anything under a top-level ``models/`` directory is skipped: that is code and
checkpoints, not data (see ``package_source.py``).

Verified Kaggle sources (slug -> role in the proposal's Table 6.1):

    ishtiaqueahmed354/metti-dataset (MEETI)                   MIMIC-IV-ECG derived
    biswajitkrshaw/mimic-iv-ecg-diagnostic-electrocardiogram  MIMIC-IV-ECG  pretrain
    ccf002/code-15-ecg-complete | gerardolug/ecg-code-15      CODE-15       pretrain
    erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database   Chapman       pretrain
    bjoernjostein/shaoxing-and-ningbo-first-hospital-database Chapman+Ningbo
    physionet/china-physiological-signal-challenge-in-2018    CPSC2018      pretrain+down
    khyeh0719/ptb-xl-dataset | omar1863/ptb-xl                PTB-XL        pretrain+down
    sahilshahare12/long-term-af-database                      LTAFDB        forward/hier
    protobioengineering/mit-bih-arrhythmia-database-modern-2023  MIT-BIH    forward/hier
"""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# key -> (regex over a directory name, human label, kind)
#
# Order matters: the first pattern that matches a directory name wins, so the more
# specific patterns (noise-stress-test before mit-bih) must come first.
DATASET_PATTERNS: List[Tuple[str, str, str, str]] = [
    # MEETI is built from MIMIC-IV-ECG (784,680 12-lead 10 s ECGs, 160,597 patients),
    # so it fills the proposal's primary large-scale pretraining slot. The Kaggle
    # dataset is spelled "metti-dataset" with the payload directory "MEETI".
    ("meeti",   r"(meeti|metti)",                              "MEETI (MIMIC-IV-ECG)", "short"),
    # nstdb is *not* the MIT-BIH Arrhythmia Database: it is 12 deliberately
    # noise-corrupted records plus three pure-noise records, and it carries no AF
    # rhythm annotations. Matched before `mitbih` so it is never mistaken for it.
    ("mitbih_nst", r"(noise[-_]?stress|nstdb)",                "MIT-BIH Noise Stress", "noise"),
    ("ptbxl",   r"ptb[-_ ]?xl",                                "PTB-XL",             "short"),
    ("cpsc",    r"(cpsc|china[-_]physiological)",              "CPSC2018",           "short"),
    ("chapman", r"(chapman|shaoxing)",                         "Chapman-Shaoxing",   "short"),
    ("ningbo",  r"ningbo",                                     "Ningbo",             "short"),
    ("code15",  r"code[-_ ]?15",                               "CODE-15",            "short"),
    ("mimic",   r"mimic",                                      "MIMIC-IV-ECG",       "short"),
    ("ltaf",    r"(long[-_ ]?term[-_ ]?af|ltafdb|ltaf)",       "Long-Term AF",       "long"),
    ("mitbih",  r"(mit[-_ ]?bih|mitdb)",                       "MIT-BIH Arrhythmia", "long"),
]

# A mirror that ships Chapman and Ningbo together
COMBINED_PATTERNS = [(r"shaoxing[-_ ]and[-_ ]ningbo",
                      [("chapman", "Chapman-Shaoxing", "short"),
                       ("ningbo", "Ningbo", "short")])]

# Directory names that never contain ECG corpora
SKIP_TOP_LEVEL = {"models", "model", "kernels", "competitions"}


@dataclass
class FoundDataset:
    key: str
    label: str
    kind: str                    # 'short' | 'long' (Holter) | 'noise'
    root: str                    # directory whose name matched
    data_root: str               # directory that actually holds the signals
    fmt: str                     # 'wfdb' | 'hdf5' | 'ptbxl' | 'mitbih_csv' | 'npy'
    n_files: int = 0
    #: Sub-directories belonging to a *different* corpus nested inside this one.
    #: Real mounts do this -- `omar1863/ptb-xl` ships the MIT-BIH Noise Stress
    #: database inside the PTB-XL mount -- and without excluding them a PTB-XL
    #: index can silently swallow noise records.
    exclude_dirs: List[str] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:      # pragma: no cover - display only
        return (f"<{self.label} kind={self.kind} fmt={self.fmt} "
                f"n={self.n_files} at {self.data_root}>")


def _match_key(name: str) -> Optional[Tuple[str, str, str]]:
    low = name.lower()
    for key, pat, label, kind in DATASET_PATTERNS:
        if re.search(pat, low):
            return key, label, kind
    return None


def _match_combined(name: str):
    low = name.lower()
    for pat, keys in COMBINED_PATTERNS:
        if re.search(pat, low):
            return keys
    return None


def _candidate_roots(input_root: str, max_depth: int = 6,
                     max_dirs: int = 20000) -> List[Tuple[str, str, str, str, int]]:
    """BFS over *directories only*, returning (key, label, kind, path, depth).

    Files are never listed here -- a corpus directory can hold hundreds of
    thousands of them, and ``os.walk`` would stat every one just to find a name.
    """
    out: List[Tuple[str, str, str, str, int]] = []
    queue: deque = deque([(input_root, 0)])
    seen_dirs = 0

    while queue and seen_dirs < max_dirs:
        path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        try:
            entries = list(os.scandir(path))
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue

        for e in entries:
            try:
                if not e.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            if e.name.startswith("."):
                continue
            if depth == 0 and e.name.lower() in SKIP_TOP_LEVEL:
                continue

            seen_dirs += 1
            combined = _match_combined(e.name)
            if combined:
                for key, label, kind in combined:
                    out.append((key, label, kind, e.path, depth + 1))
                continue

            m = _match_key(e.name)
            if m:
                key, label, kind = m
                out.append((key, label, kind, e.path, depth + 1))
                # keep descending: a mount can nest several corpora (e.g. a
                # "ptb-xl" dataset that also ships the noise-stress database)
            queue.append((e.path, depth + 1))

    return out


def _walk_probe(root: str, max_files: int = 200_000,
                exclude_dirs: Sequence[str] = ()) -> Dict[str, object]:
    """Single bounded walk collecting the signals needed to classify a corpus."""
    counts = {"hea": 0, "mat": 0, "dat": 0, "hdf5": 0, "npy": 0, "csv": 0, "atr": 0}
    first: Dict[str, str] = {}
    ptbxl_csv: Optional[str] = None
    mitbih_csv = 0
    seen = 0
    excluded = {os.path.abspath(p) for p in exclude_dirs}

    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.abspath(dirpath) in excluded:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                       and os.path.abspath(os.path.join(dirpath, d)) not in excluded]
        for fn in filenames:
            seen += 1
            if seen > max_files:
                break
            low = fn.lower()
            if low == "ptbxl_database.csv":
                ptbxl_csv = os.path.join(dirpath, fn)
            if low.endswith("_ekg.csv"):
                mitbih_csv += 1
            for ext in counts:
                if low.endswith("." + ext):
                    counts[ext] += 1
                    first.setdefault(ext, os.path.join(dirpath, fn))
                    break
        if seen > max_files:
            break

    return {"counts": counts, "first": first, "ptbxl_csv": ptbxl_csv,
            "mitbih_csv": mitbih_csv, "truncated": seen > max_files}


def _classify_format(probe: Dict[str, object]) -> Optional[str]:
    counts = probe["counts"]                       # type: ignore[index]
    if probe["ptbxl_csv"]:                         # type: ignore[index]
        return "ptbxl"
    if probe["mitbih_csv"]:                        # type: ignore[index]
        return "mitbih_csv"
    if counts["hdf5"] > 0:
        return "hdf5"
    if counts["hea"] > 0:
        return "wfdb"
    if counts["npy"] > 0:
        return "npy"
    return None


def _data_root(probe: Dict[str, object], fmt: str, mount: str) -> str:
    first = probe["first"]                         # type: ignore[index]
    if fmt == "ptbxl":
        return os.path.dirname(str(probe["ptbxl_csv"]))    # type: ignore[index]
    if fmt == "hdf5" and "hdf5" in first:
        return os.path.dirname(first["hdf5"])
    if fmt == "npy" and "npy" in first:
        return os.path.dirname(first["npy"])
    return mount                                   # loaders walk recursively


def discover(input_root: str = "/kaggle/input",
             only: Optional[Sequence[str]] = None,
             verbose: bool = True,
             max_depth: int = 6) -> Dict[str, FoundDataset]:
    """Return {key: FoundDataset} for every recognised corpus under ``input_root``."""
    found: Dict[str, FoundDataset] = {}
    if not os.path.isdir(input_root):
        if verbose:
            print(f"[discovery] {input_root} does not exist - nothing mounted.")
        return found

    candidates = _candidate_roots(input_root, max_depth=max_depth)
    # shallowest match wins for each corpus key
    candidates.sort(key=lambda c: (c[0], c[4], c[3]))

    # One root per corpus key, resolved before probing so that a corpus nested inside
    # another (PTB-XL mounts that also ship the noise-stress database) can be excluded
    # from its parent's probe rather than polluting the parent's classification.
    chosen: Dict[str, Tuple[str, str, str]] = {}
    for key, label, kind, path, _depth in candidates:
        if key in chosen or (only is not None and key not in only):
            continue
        chosen[key] = (label, kind, path)

    all_roots = [p for _l, _k, p in chosen.values()]

    for key, (label, kind, path) in chosen.items():
        nested = [r for r in all_roots
                  if r != path and os.path.abspath(r).startswith(
                      os.path.abspath(path) + os.sep)]
        probe = _walk_probe(path, exclude_dirs=nested)
        fmt = _classify_format(probe)
        if fmt is None:
            if verbose:
                print(f"[discovery] {label}: no readable ECG files under {path}, skipping")
            continue

        counts = probe["counts"]                   # type: ignore[index]
        n_files = {"ptbxl": counts["dat"] or counts["hea"],
                   "wfdb": counts["hea"],
                   "hdf5": counts["hdf5"],
                   "npy": counts["npy"],
                   "mitbih_csv": probe["mitbih_csv"]}[fmt]   # type: ignore[index]

        found[key] = FoundDataset(key=key, label=label, kind=kind, root=path,
                                  data_root=_data_root(probe, fmt, path), fmt=fmt,
                                  n_files=int(n_files), exclude_dirs=nested)
        if verbose:
            approx = "+" if probe.get("truncated") else ""
            print(f"[discovery] {label:24s} fmt={fmt:10s} files~{n_files}{approx:<2s} {path}")
            for n in nested:
                print(f"             excluding nested corpus: {os.path.basename(n)}")

    if verbose:
        if not found:
            print(f"[discovery] no recognised ECG datasets under {input_root}.")
        _warn_about_gaps(found)
    return found


def _warn_about_gaps(found: Dict[str, FoundDataset]):
    """Say plainly which experiments the attached data cannot support."""
    has_long = any(d.kind == "long" for d in found.values())
    if not has_long:
        print("\n[discovery] NOTE: no long-recording corpus with rhythm annotations "
              "(Long-Term AF / MIT-BIH Arrhythmia) was found.")
        print("            Without one there is no real 'future window' and no AF-onset "
              "labels, so:")
        print("              - L_fwd trains only via the intra-window causal path;")
        print("              - Stage 2 (anticipation) and Stage 4 (hierarchical) fall "
              "back to synthetic data.")
        print("            Attach `sahilshahare12/long-term-af-database` and/or "
              "`protobioengineering/mit-bih-arrhythmia-database-modern-2023` to enable them.")
    if "mitbih_nst" in found and "mitbih" not in found:
        print("\n[discovery] NOTE: the MIT-BIH *Noise Stress Test* database was found. "
              "It is deliberately noise-corrupted and has no AF rhythm annotations, so "
              "it is not used as an anticipation corpus. Enable it for pretraining with "
              "cfg.data.use_noise_corpora = True.")


def summarize(found: Dict[str, FoundDataset]) -> str:
    if not found:
        return "No ECG datasets discovered."
    lines = [f"{'key':12s} {'dataset':24s} {'kind':6s} {'fmt':11s} {'files':>9s}",
             "-" * 68]
    for k, d in sorted(found.items()):
        lines.append(f"{k:12s} {d.label:24s} {d.kind:6s} {d.fmt:11s} {d.n_files:>9d}")
    return "\n".join(lines)
