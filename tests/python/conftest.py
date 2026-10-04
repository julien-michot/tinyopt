# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""Use the installed `tinyopt` if any, else the source tree plus the locally built C library."""

import glob
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if importlib.util.find_spec("tinyopt") is None:
    sys.path.insert(0, str(ROOT / "bindings" / "python"))
    if "TINYOPT_C_LIBRARY" not in os.environ:
        hits = sorted(glob.glob(str(ROOT / "build-*" / "*tinyopt_c.*")) +
                      glob.glob(str(ROOT / "build-*" / "*" / "*tinyopt_c.*")))
        hits = [h for h in hits if h.endswith((".so", ".dylib", ".dll"))]
        # Prefer the dedicated python build dir.
        hits.sort(key=lambda h: "build-python" not in h)
        if hits:
            os.environ["TINYOPT_C_LIBRARY"] = hits[0]
        else:
            import pytest

            pytest.exit("libtinyopt_c not found: run `pixi run build-python` once", returncode=2)
