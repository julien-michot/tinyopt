#!/usr/bin/env python3
import functools
import numpy as np
import tinyopt


def test_plain_residual():
    def res_fn(x):
        return x - np.array([5.0, 3.0])

    opts = tinyopt.Options()
    x0 = np.array([3.0, 2.0])
    xf, out = tinyopt.optimize(x0, res_fn, opts=opts)
    assert out.stop_reason >= 0


def test_cost_and_grad():
    def res_fn(x, g):
        res = x - np.array([5.0, 3.0])
        if g is not None:
            g[:] = 2.0 * res
        return (res.T * res)[0]

    opts = tinyopt.Options()
    opts.solver_type = tinyopt.Solver.GradientDescent
    x0 = np.array([3.0, 2.0])
    xf, out = tinyopt.optimize(x0, res_fn, opts=opts)
    assert out.stop_reason >= 0


def test_decorated_and_bound():
    def deco(f):
        @functools.wraps(f)
        def w(*a, **k):
            return f(*a, **k)
        return w

    def res_fn(x):
        return x - np.array([5.0, 3.0])

    wrapped = deco(res_fn)

    class C:
        def __call__(self, x):
            return x - np.array([5.0, 3.0])

    inst = C()

    opts = tinyopt.Options()
    x0 = np.array([3.0, 2.0])

    xf1, out1 = tinyopt.optimize(x0, wrapped, opts=opts)
    xf2, out2 = tinyopt.optimize(x0, inst, opts=opts)
    assert out1.stop_reason >= 0
    assert out2.stop_reason >= 0
