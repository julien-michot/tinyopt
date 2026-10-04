# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""Fit a circle (center, radius) to noisy 2D points, three ways to provide derivatives."""

import numpy as np

import tinyopt

rng = np.random.default_rng(0)
angles = rng.uniform(0, 2 * np.pi, 200)
truth = np.array([1.5, -0.5, 2.0])  # cx, cy, r
pts = truth[:2] + truth[2] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
pts += rng.normal(scale=0.01, size=pts.shape)


def residuals(p):  # finite-difference Jacobian, nothing else to write
    return np.linalg.norm(pts - p[:2], axis=1) - p[2]


def residuals_jac(p):  # return (r, J): J is (200, 3)
    d = pts - p[:2]
    dist = np.linalg.norm(d, axis=1)
    return dist - p[2], np.column_stack([-d / dist[:, None], -np.ones(len(pts))])


def residuals_inplace(p, jac):  # like the C API: fill `jac` in place, None when not needed
    d = pts - p[:2]
    dist = np.linalg.norm(d, axis=1)
    if jac is not None:
        jac[:, :2] = -d / dist[:, None]
        jac[:, 2] = -1.0
    return dist - p[2]


for fn in (residuals, residuals_jac, residuals_inplace):
    res = tinyopt.optimize(fn, [0.0, 0.0, 1.0])  # 3 parameters: fixed-size solver is automatic
    print(f"{fn.__name__:18s} -> {np.round(res.x, 3)} in {res.iters} its, "
          f"numerical diff: {res.numerical_diff}")
    np.testing.assert_allclose(res.x, truth, atol=0.02)
