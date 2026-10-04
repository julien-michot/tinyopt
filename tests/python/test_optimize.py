# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import re
from pathlib import Path

import numpy as np
import pytest

import tinyopt
from tinyopt import _capi

TARGET = np.array([1.0, 2.0, 3.0])
NO_LOG = {"log_enabled": False}
FIXED = [True, False]


def test_scalar_sqrt2_all_precisions_paths():
    for fixed in (None, False):
        res = tinyopt.optimize(lambda x: x * x - 2.0, 1.0, fixed=fixed, **NO_LOG)
        assert isinstance(res.x, float)
        assert res.x == pytest.approx(2 ** 0.5, abs=1e-6)
        assert res.success and res.converged and res.numerical_diff


@pytest.mark.parametrize("fixed", FIXED)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_residuals_numdiff_and_analytic(fixed, dtype):
    tol = 1e-4 if dtype is np.float32 else 1e-6
    x0 = np.zeros(3, dtype=dtype)
    res = tinyopt.optimize(lambda x: x - TARGET, x0, fixed=fixed, **NO_LOG)
    assert res.x.dtype == dtype and res.numerical_diff
    np.testing.assert_allclose(res.x, TARGET, atol=tol)
    np.testing.assert_array_equal(x0, 0)  # the initial guess is never modified

    res = tinyopt.optimize(lambda x: (x - TARGET, np.eye(3)), x0, fixed=fixed, **NO_LOG)
    assert not res.numerical_diff and res.converged
    np.testing.assert_allclose(res.x, TARGET, atol=tol)


def test_rosenbrock_residuals_dynamic_and_fixed_agree():
    def f(x):
        return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])

    def J(x):
        return f(x), np.array([[-20 * x[0], 10.0], [-1.0, 0.0]])

    for fn in (f, J):
        a = tinyopt.optimize(fn, [-1.2, 1.0], fixed=True, max_iters=100, **NO_LOG)
        b = tinyopt.optimize(fn, [-1.2, 1.0], fixed=False, max_iters=100, **NO_LOG)
        np.testing.assert_allclose(a.x, [1, 1], atol=1e-5)
        np.testing.assert_allclose(b.x, a.x, atol=1e-8)


def _rosenbrock(x):
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def _rosenbrock_jacobian(x):
    return np.array([[-20 * x[0], 10.0], [-1.0, 0.0]])


def test_analytic_jacobian_matches_finite_differences():
    x = np.array([-1.2, 1.0])
    eps = 1e-6
    numeric = np.stack([(_rosenbrock(x + eps * e) - _rosenbrock(x - eps * e)) / (2 * eps)
                        for e in np.eye(2)], axis=1)
    np.testing.assert_allclose(_rosenbrock_jacobian(x), numeric, atol=1e-5)


@pytest.mark.parametrize("fixed", FIXED)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_inplace_jacobian_like_c_api(fixed, dtype):
    requests = {"with": 0, "without": 0}

    def f(x, jac):
        if jac is None:  # Tinyopt only needs residuals (e.g. trial steps)
            requests["without"] += 1
        else:
            requests["with"] += 1
            assert jac.shape == (2, 2) and jac.flags.writeable and jac.dtype == dtype
            jac[...] = _rosenbrock_jacobian(x)
        return _rosenbrock(x)

    x0 = np.array([-1.2, 1.0], dtype=dtype)
    res = tinyopt.optimize(f, x0, fixed=fixed, max_iters=100, **NO_LOG)
    assert not res.numerical_diff
    if dtype is np.float64:  # float32 Rosenbrock may stall; it must still match the tuple style
        np.testing.assert_allclose(res.x, [1, 1], atol=1e-5)
    assert requests["with"] > 0 and requests["without"] > 0

    ref = tinyopt.optimize(lambda x: (_rosenbrock(x), _rosenbrock_jacobian(x)), x0,
                           fixed=fixed, max_iters=100, **NO_LOG)
    np.testing.assert_allclose(res.x, ref.x, atol=1e-5)


def test_inplace_jacobian_rectangular_and_scalar():
    t = np.linspace(0, 1, 20)

    def line(p, jac):
        if jac is not None:
            jac[:, 0] = t
            jac[:, 1] = 1.0
        return p[0] * t + p[1] - (2.0 * t + 0.5)

    res = tinyopt.optimize(line, [0.0, 0.0], **NO_LOG)
    np.testing.assert_allclose(res.x, [2.0, 0.5], atol=1e-6)

    def sqr(x, jac):
        assert isinstance(x, float)
        if jac is not None:
            jac[0, 0] = 2 * x
        return x * x - 2.0

    res = tinyopt.optimize(sqr, 1.0, **NO_LOG)
    assert res.x == pytest.approx(2 ** 0.5, abs=1e-6) and not res.numerical_diff


def test_large_dynamic_problem_has_no_fixed_solver():
    n = 25
    t = np.linspace(0, 1, n)
    res = tinyopt.optimize(lambda x: (x - t, np.eye(n)), np.zeros(n), **NO_LOG)
    np.testing.assert_allclose(res.x, t, atol=1e-6)
    with pytest.raises(ValueError, match="no fixed-size"):
        tinyopt.optimize(lambda x: x - t, np.zeros(n), fixed=True, **NO_LOG)


def test_more_residuals_than_parameters_line_fit():
    t = np.linspace(0, 1, 20)
    y = 2.0 * t + 0.5

    def f(p):
        r = p[0] * t + p[1] - y
        return r, np.stack([t, np.ones_like(t)], axis=1)

    res = tinyopt.optimize(f, [0.0, 0.0], **NO_LOG)
    np.testing.assert_allclose(res.x, [2.0, 0.5], atol=1e-6)
    assert res.num_residuals == 20


@pytest.mark.parametrize("fixed", FIXED)
def test_cost_problem(fixed):
    res = tinyopt.optimize(lambda x: float(((x - TARGET) ** 2).sum()), np.zeros(3),
                           "cost", fixed=fixed, **NO_LOG)
    np.testing.assert_allclose(res.x, TARGET, atol=1e-4)


@pytest.mark.parametrize("fixed", FIXED)
@pytest.mark.parametrize("solver", ["bfgs", "lbfgs", "gd"])
def test_gradient_problem(fixed, solver):
    def f(x, g):
        d = x - TARGET
        if g is not None:
            g += d
        return 0.5 * d @ d

    res = tinyopt.optimize(f, np.zeros(3), "gradient", solver=solver, fixed=fixed,
                           max_iters=500, gd_learning_rate=0.5, **NO_LOG)
    np.testing.assert_allclose(res.x, TARGET, atol=1e-3)


@pytest.mark.parametrize("fixed", FIXED)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_hessian_problem_is_accumulated_in_place(fixed, dtype):
    A = np.diag([1.0, 2.0, 4.0])
    seen = {}

    def f(x, g, H):
        d = x - TARGET
        if g is not None:
            g += A @ d
        if H is not None:
            H += A
            seen["H_shape"] = H.shape
        return 0.5 * d @ A @ d

    res = tinyopt.optimize(f, np.zeros(3, dtype=dtype), "hessian", fixed=fixed, **NO_LOG)
    np.testing.assert_allclose(res.x, TARGET, atol=1e-4)
    assert seen["H_shape"] == (3, 3)


def test_scalar_accumulation_uses_one_element_buffers():
    def f(x, g, H):
        assert isinstance(x, float)
        if g is not None:
            g += 2 * (x - 3.0)
        if H is not None:
            H += 2.0
        return (x - 3.0) ** 2

    res = tinyopt.optimize(f, 0.0, "hessian", **NO_LOG)
    assert res.x == pytest.approx(3.0, abs=1e-5)


def test_input_view_is_zero_copy_and_read_only():
    x0 = np.zeros(3)
    ptrs, writable = set(), []

    def f(x):
        ptrs.add(x.__array_interface__["data"][0])
        writable.append(x.flags.writeable)
        return x - TARGET

    tinyopt.optimize(f, x0, fixed=False, **NO_LOG)
    assert not any(writable)
    assert len(ptrs) <= 8  # views alias Tinyopt's few internal buffers, no per-call copies


def test_exceptions_propagate_and_stop():
    calls = []

    def f(x):
        calls.append(1)
        if len(calls) > 2:
            raise KeyError("boom")
        return x - TARGET

    with pytest.raises(KeyError, match="boom"):
        tinyopt.optimize(f, np.zeros(3), **NO_LOG)


def test_bad_return_sizes_raise():
    calls = []

    def shrinking(x):
        calls.append(1)
        return np.ones(3 if len(calls) == 1 else 2)

    with pytest.raises(ValueError, match="residuals"):
        tinyopt.optimize(shrinking, np.zeros(3), **NO_LOG)
    with pytest.raises(ValueError, match="Jacobian"):
        tinyopt.optimize(lambda x: (x - TARGET, np.eye(2)), np.zeros(3), **NO_LOG)


def test_stop_callback_and_plus_eq():
    seen = []
    res = tinyopt.optimize(lambda x: x - TARGET, np.zeros(3),
                           stop_callback=lambda e, s, g: seen.append(e) or True, **NO_LOG)
    assert len(seen) == 1 and res.success

    calls = []

    def plus_eq(x, dx):
        calls.append(1)
        x += dx

    res = tinyopt.optimize(lambda x: (x - TARGET, np.eye(3)), np.zeros(3),
                           plus_eq=plus_eq, max_iters=200, **NO_LOG)
    assert calls
    np.testing.assert_allclose(res.x, TARGET, atol=1e-4)


def test_solver_options_and_errors():
    for solver in ("lm", "gn", "dogleg"):
        res = tinyopt.optimize(lambda x: x - TARGET, np.zeros(3), solver=solver,
                               max_iters=500, **NO_LOG)
        assert res.success, solver
    with pytest.raises(ValueError, match="scalar objective"):
        tinyopt.optimize(lambda x: x - TARGET, np.zeros(3), solver="bfgs")
    for ls in ("ldlt", "llt", "lu", "qr", "svd"):
        res = tinyopt.optimize(lambda x: x - TARGET, np.zeros(3), "residuals",
                               linear_solver=ls, solver="gn", **NO_LOG)
        np.testing.assert_allclose(res.x, TARGET, atol=1e-5)
    with pytest.raises(ValueError, match="solver"):
        tinyopt.optimize(lambda x: x, 1.0, solver="nope")
    with pytest.raises(TypeError, match="unexpected option"):
        tinyopt.optimize(lambda x: x, 1.0, bogus=1)
    with pytest.raises(ValueError, match="max_iters"):
        tinyopt.optimize(lambda x: x, 1.0, max_iters=100000)
    with pytest.raises(ValueError, match="problem"):
        tinyopt.optimize(lambda x: x, 1.0, "nope")
    with pytest.raises(ValueError, match="1-D"):
        tinyopt.optimize(lambda x: x, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="empty"):
        tinyopt.optimize(lambda x: x, [])


def test_max_iters_option_is_honored():
    res = tinyopt.optimize(lambda x: x * x - 2.0, 100.0, max_iters=2, **NO_LOG)
    assert res.stop_reason == tinyopt.StopReason.MAX_ITERS


def test_python_list_and_int_initial_guess():
    res = tinyopt.optimize(lambda x: x - TARGET, [0, 0, 0], **NO_LOG)
    np.testing.assert_allclose(res.x, TARGET, atol=1e-5)
    res = tinyopt.optimize(lambda x: x - 3.0, 0, **NO_LOG)
    assert res.x == pytest.approx(3.0, abs=1e-5)


def test_options_layout_matches_header():
    header = Path(__file__).resolve().parents[2] / "include/tinyopt/c/c_api_common.h"
    if not header.is_file():
        pytest.skip("C headers not available")
    body = header.read_text().split("typedef struct tinyopt_options_t {")[1].split("}")[0]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    expected = []
    for decl in body.split(";"):
        decl = decl.strip()
        if decl:
            expected.append(re.split(r"[\s*]+", decl)[-1])
    assert [n for n, _ in _capi.Options._fields_] == expected


def test_summary_and_options_defaults():
    opts = _capi.default_options()
    assert opts.max_iters == 50 and opts.solver_type == _capi.SOLVERS["lm"]
    assert ctypes.sizeof(_capi.Summary) == 32
