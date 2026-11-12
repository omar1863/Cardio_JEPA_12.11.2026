#!/usr/bin/env python3
"""Verify the shipped notebook and the Kaggle-Model export stay consistent.

The notebook no longer embeds the package source, so the drift risk moved: it is now
possible to add a module to `cardio_jepa/` and forget that the uploaded Kaggle Model
still holds the old tree. This checks:

  1. the notebook contains no leftover `%%writefile` cells;
  2. the notebook's bootstrap cell is present and references a model input;
  3. `tools/export_kaggle_model.py` copies every `.py` under `cardio_jepa/`, and the
     exported tree imports on its own.

    python tools/check_export.py        # exit 1 on any problem
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(REPO, "notebooks", "cardio_jepa_kaggle.ipynb")


def package_modules() -> set:
    out = set()
    for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, "cardio_jepa")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".py"):
                out.add(os.path.relpath(os.path.join(dirpath, fn),
                                        os.path.join(REPO, "cardio_jepa")))
    return out


def main() -> int:
    problems = []

    if not os.path.exists(NB):
        print(f"FAIL: {NB} missing - run tools/build_notebook.py")
        return 1

    with open(NB) as fh:
        nb = json.load(fh)
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    stale = [s.splitlines()[0] for s in sources if s.startswith("%%writefile")]
    if stale:
        problems.append(f"{len(stale)} leftover %%writefile cell(s): {stale[:3]}")
    if not any("find_package_root" in s for s in sources):
        problems.append("bootstrap cell missing (no find_package_root)")
    if not any("/kaggle/input/models" in s for s in sources):
        problems.append("bootstrap cell does not reference a Kaggle Model input")

    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable,
                            os.path.join(REPO, "tools", "export_kaggle_model.py"),
                            "-o", tmp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(f"export_kaggle_model.py failed:\n{r.stdout}\n{r.stderr}")
        else:
            exported = set()
            base = os.path.join(tmp, "cardio_jepa")
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in filenames:
                    if fn.endswith(".py"):
                        exported.add(os.path.relpath(os.path.join(dirpath, fn), base))
            missing = package_modules() - exported
            if missing:
                problems.append(f"export is missing {len(missing)} module(s): "
                                f"{sorted(missing)[:5]}")
            if "import check: OK" not in r.stdout:
                problems.append("exported tree failed its import check")

    if problems:
        print("Problems found:")
        for p in problems:
            print("  -", p)
        return 1

    print(f"OK: notebook has no embedded source; export covers all "
          f"{len(package_modules())} modules and imports cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
