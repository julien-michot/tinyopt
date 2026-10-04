# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""Accumulation: build the Hessian J^T J and gradient J^T r by hand over many samples.

The gradient/Hessian are zero-copy views of Tinyopt's buffers (no matrix is allocated or
returned), and this works for any number of parameters, even above the fixed-size solvers.
"""

import numpy as np

import tinyopt

rng = np.random.default_rng(1)
n = 20
true_w = rng.normal(size=n)
X = rng.normal(size=(500, n))
y = X @ true_w


def accumulate(w, grad, hess):
    r = X @ w - y
    if grad is not None:
        grad += X.T @ r
    if hess is not None:
        hess += X.T @ X
    return 0.5 * r @ r


res = tinyopt.optimize(accumulate, np.zeros(n), "hessian")  # n=20: dynamic-size solver
print(f"{n} weights recovered, max error {np.abs(res.x - true_w).max():.2e}")
assert np.allclose(res.x, true_w, atol=1e-5)
