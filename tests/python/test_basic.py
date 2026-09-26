#!/usr/bin/env python3
import math
import numpy as np

import tinyopt


def test_nlls_simple():
    # residuals: res = x - [5, 3]
    def res_fn(x):
        return x - np.array([5.0, 3.0])

    # Exercise nested Options.log flag via attribute-style assignment when possible
    opts = tinyopt.Options()
    opts.log.print_x = True

    x0 = np.array([3.0, 2.0])
    xf, out = tinyopt.optimize(x0, res_fn, opts=opts)
    assert out.stop_reason >= 0
    # TODO check converged to correct solution

    # final x should be closer to target than initial
    assert abs(xf[0] - 5.0) <= abs(x0[0] - 5.0)
    assert abs(xf[1] - 3.0) <= abs(x0[1] - 3.0)


def test_nlls_simple_grad():

    def res_fn(x, grad):
        res = x - np.array([5.0, 3.0])
        if grad is not None:
            grad[:] = 2.0 * res  # gradient of squared residuals, alway use [:] to modify in-place!!
        return res @ res.T # return squared residuals


    x0 = np.array([3.0, 2.0])
    assert tinyopt.check_gradient(x0, res_fn) # verify gradient is correct

    # Exercise nested Options.log flag via attribute-style assignment when possible
    opts = tinyopt.Options()
    opts.solver_type = tinyopt.Solver.GradientDescent
    opts.gd.lr = 0.1
    opts.log.print_x = True

    xf, out = tinyopt.optimize(x0, res_fn, opts=opts)
    assert out.stop_reason >= 0
    # TODO check converged to correct solution

    # final x should be closer to target than initial
    assert abs(xf[0] - 5.0) <= abs(x0[0] - 5.0)
    assert abs(xf[1] - 3.0) <= abs(x0[1] - 3.0)

if __name__ == "__main__":
    test_nlls_simple()
    test_nlls_simple_grad()
