# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""Tinyopt: fast non-linear least squares and unconstrained optimization.

    import tinyopt
    res = tinyopt.optimize(lambda x: x * x - 2.0, 1.0)  # sqrt(2)
"""

from .core import Result, StopReason, optimize

try:
    from importlib.metadata import version as _version

    __version__ = _version("tinyopt")
except Exception:  # running from a source tree
    __version__ = "0.7.0"

__all__ = ["optimize", "Result", "StopReason", "__version__"]
