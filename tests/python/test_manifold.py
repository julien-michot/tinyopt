#!/usr/bin/env python3
import math
import numpy as np

import tinyopt


def test_manifold_angle_wrap():
    # Treat x as an angle that should wrap to the range [-pi, pi]
    def res_fn(x):
        # target angle is pi/2
        target = np.array([math.pi / 2.0])
        # compute difference and wrap to [-pi, pi]
        d = x - target
        d = (d + math.pi) % (2.0 * math.pi) - math.pi
        return d

    x0 = np.array([3.0])  # ~171 deg, target is 90 deg

    # Provide a plus_eq callback that applies the manifold-aware update.
    def plus(x_arr, dx_arr):
        return (x_arr + dx_arr + math.pi) % (2.0 * math.pi) - math.pi

    # Use Options and set solver type explicitly to LevenbergMarquardt (attribute-style)
    opts = tinyopt.Options()
    opts.solver_type = tinyopt.Solver.LevenbergMarquardt
    xf, out = tinyopt.optimize(x0, res_fn, plus, opts)

    # Ensure enum write works (attribute-style)
    out.stop_reason = tinyopt.StopReason.kNone
    assert out.stop_reason == tinyopt.StopReason.kNone

    # final x should be closer to target than initial
    initial_diff = abs(((x0[0] - math.pi/2.0 + math.pi)%(2*math.pi)) - math.pi)
    final_diff = abs(((xf[0] - math.pi/2.0 + math.pi)%(2*math.pi)) - math.pi)
    assert final_diff <= initial_diff


if __name__ == "__main__":
    test_manifold_angle_wrap()
    print("OK")
