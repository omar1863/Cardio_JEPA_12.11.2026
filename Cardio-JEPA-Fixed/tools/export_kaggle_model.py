#!/usr/bin/env python3
"""Package `cardio_jepa` into the directory layout you upload as a Kaggle Model.

The notebook imports the package from a Kaggle Model input rather than writing the
source out itself, so the package has to get there somehow. Run this, then upload
the produced directory at https://www.kaggle.com/models -> New Model.

    python tools/export_kaggle_model.py                 # -> build/kaggle_model/
    python tools/export_kaggle_model.py --zip           # also make a .zip to upload
    python tools/export_kaggle_model.py --checkpoint runs/dual.pth

Upload the *contents* of the output directory. With Framework=PyTorch,
Variation=default, Version=1 and a model slug of `cardio-jepa`, Kaggle mounts it at

    /kaggle/input/models/<your-username>/cardio-jepa/pytorch/default/1/

which is the path the notebook looks for first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}

README = """# Cardio-JEPA package

Source for the `cardio_jepa` package, uploaded as a Kaggle Model so notebooks can
import it without pip, an internet connection, or `%%writefile` cells.

Usage inside a Kaggle notebook:

```python
import sys
sys.path.insert(0, "/kaggle/input/models/<owner>/cardio-jepa/pytorch/default/1")
import cardio_jepa as cj
```

Any `.pth` files alongside it are pretraining checkpoints and can be used to
warm-start a run:

```python
cj.load_weights(model, "<path to .pth>")
```

Regenerate with `python tools/export_kaggle_model.py` from the repository.
"""


def copy_package(dst: str) -> int:
    src = os.path.join(REPO, "cardio_jepa")
    out = os.path.join(dst, "cardio_jepa")
    if os.path.exists(out):
        shutil.rmtree(out)
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel = os.path.relpath(dirpath, src)
        target = out if rel == "." else os.path.join(out, rel)
        os.makedirs(target, exist_ok=True)
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            shutil.copy2(os.path.join(dirpath, fn), os.path.join(target, fn))
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join(REPO, "build", "kaggle_model"))
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="checkpoint file(s) to bundle alongside the package")
    ap.add_argument("--zip", action="store_true", help="also produce a .zip archive")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = copy_package(args.out)
    with open(os.path.join(args.out, "README.md"), "w") as fh:
        fh.write(README)

    for ck in args.checkpoint:
        if not os.path.isfile(ck):
            print(f"  ! checkpoint not found, skipping: {ck}")
            continue
        shutil.copy2(ck, os.path.join(args.out, os.path.basename(ck)))
        print(f"  + {os.path.basename(ck)} "
              f"({os.path.getsize(ck) / 1e6:.1f} MB)")

    print(f"wrote {n} modules to {args.out}")

    # sanity: the exported tree must be importable on its own
    sys.path.insert(0, args.out)
    for mod in list(sys.modules):
        if mod == "cardio_jepa" or mod.startswith("cardio_jepa."):
            del sys.modules[mod]
    try:
        import cardio_jepa  # noqa: F401
        print(f"import check: OK (cardio_jepa {cardio_jepa.__version__} from "
              f"{os.path.dirname(cardio_jepa.__file__)})")
    except Exception as exc:
        print(f"import check FAILED: {type(exc).__name__}: {exc}")
        return 1

    if args.zip:
        archive = shutil.make_archive(args.out, "zip", args.out)
        print(f"archive: {archive} ({os.path.getsize(archive) / 1e6:.1f} MB)")

    print("\nNext: upload the contents of this directory at "
          "https://www.kaggle.com/models -> New Model")
    print("  Framework=PyTorch  Variation=default  Version=1  slug=cardio-jepa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
