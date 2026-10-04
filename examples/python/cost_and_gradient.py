# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""Scalar objectives: derivative-free cost, and manual gradient accumulation."""

import numpy as np

import tinyopt

target = np.array([1.0, 2.0, 3.0])

# problem="cost": only the value is needed, derivatives are estimated.
res = tinyopt.optimize(lambda x: float(((x - target) ** 2).sum()), np.zeros(3), "cost")
print("cost    ->", np.round(res.x, 4))
assert np.allclose(res.x, target, atol=1e-3)


# problem="gradient": accumulate into the (zeroed) gradient view, in place and copy-free.
def cost_and_gradient(x, grad):
    d = x - target
    if grad is not None:  # None when Tinyopt only needs the cost
        grad += d
    return 0.5 * d @ d


res = tinyopt.optimize(cost_and_gradient, np.zeros(3), "gradient", solver="lbfgs")
print("gradient ->", np.round(res.x, 4))
assert np.allclose(res.x, target, atol=1e-3)
