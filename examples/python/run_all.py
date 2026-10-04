# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""Runs every Python example (used by `pixi run examples-python`)."""

import importlib.util
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

if importlib.util.find_spec("tinyopt") is None:  # not installed: use the source tree
    sys.path.insert(0, str(ROOT / "bindings" / "python"))
    for lib in sorted((ROOT / "build-python").glob("*tinyopt_c.*")):
        os.environ.setdefault("TINYOPT_C_LIBRARY", str(lib))

for example in sorted(HERE.glob("*.py")):
    if example.name != Path(__file__).name:
        print(f"== {example.name}")
        runpy.run_path(str(example), run_name="__main__")
