# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""High-level `optimize()` entry point built on the Tinyopt C API."""

import ctypes as C
import enum
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from . import _capi as capi

_PROBLEMS = {
    "cost": capi.EVAL_COST,
    "residuals": capi.EVAL_RESIDUALS,
    "gradient": capi.EVAL_GRADIENT,
    "hessian": capi.EVAL_HESSIAN,
}
_OPTION_FIELDS = dict(capi.Options._fields_)
_FIRST_ORDER = {"gd", "cg", "bfgs", "lbfgs"}
_RESERVED_FIELDS = {"stop_callback", "stop_callback_user_data", "stop_callback2",
                    "stop_callback2_user_data", "solver_type", "linear_solver"}
_INT_LIMITS = {C.c_ubyte: 1 << 8, C.c_ushort: 1 << 16}


class StopReason(enum.IntEnum):
    """Why the optimization stopped (mirrors `tinyopt::StopReason`)."""

    OUT_OF_MEMORY = -4
    SOLVER_FAILED = -3
    NAN_OR_INF = -2
    SKIPPED = -1
    NONE = 0
    MIN_ERROR = 1
    MIN_RELATIVE_ERROR = 2
    MIN_STEP = 3
    MIN_GRADIENT = 4
    MAX_ITERS = 5
    MAX_FAILURES = 6
    MAX_CONSECUTIVE_FAILURES = 7
    TIMED_OUT = 8
    USER_STOPPED = 9


@dataclass
class Result:
    """Outcome of `optimize()`. `x` has the type of the initial guess (float or ndarray)."""

    x: Any
    cost: float
    iters: int
    failures: int
    stop_reason: StopReason
    success: bool
    num_residuals: int
    numerical_diff: bool

    @property
    def converged(self) -> bool:
        """True if a convergence criterion (error, relative error, step, gradient) was met."""
        return StopReason.MIN_ERROR <= self.stop_reason <= StopReason.MIN_GRADIENT


def _viewer(ct, npdt, shape, order="C", writable=True):
    """Zero-copy numpy view factory over raw C memory, cached by address."""
    cache = {}
    array_type = ct * int(np.prod(shape))

    def get(addr):
        view = cache.get(addr)
        if view is None:
            if len(cache) >= 16:
                cache.clear()
            view = np.frombuffer(array_type.from_address(addr), dtype=npdt)
            view = view.reshape(shape, order=order)
            view.flags.writeable = writable
            cache[addr] = view
        return view

    return get


def _build_callback(problem, fn, n, m, analytic, inplace_jac, scalar, ct, npdt, errors):
    cb_type = capi.callback_types(ct)[_PROBLEMS[problem]]
    xview = _viewer(ct, npdt, (n,), writable=False)
    xin = (lambda addr: float(xview(addr)[0])) if scalar else xview

    if problem == "cost":

        def cb(x, dims, cost, user):
            try:
                cost[0] = fn(xin(x))
                return 0
            except BaseException as e:  # incl. KeyboardInterrupt: stop, then re-raise
                errors.append(e)
                return 1

    elif problem == "residuals":
        rview = _viewer(ct, npdt, (m,))
        jview = _viewer(ct, npdt, (m, n))

        def cb(x, dims, res, jac, num_res, user):
            try:
                if inplace_jac:
                    jptr = jac[0]
                    out = fn(xin(x), None if jptr is None else jview(jptr))
                else:
                    out = fn(xin(x))
                    if analytic:
                        if type(out) is not tuple or len(out) != 2:
                            raise ValueError("residuals must consistently return (r, J)")
                        out, J = out
                        jptr = jac[0]
                        if jptr is not None:
                            J = np.asarray(J, dtype=npdt)
                            if J.size != m * n:
                                raise ValueError(
                                    f"Jacobian has {J.size} values, expected {m}x{n}")
                            jview(jptr)[...] = J.reshape(m, n)
                    else:
                        jac[0] = None  # let Tinyopt estimate the Jacobian
                r = np.asarray(out, dtype=npdt)
                if r.size != m:
                    raise ValueError(f"function returned {r.size} residuals, expected {m}")
                rview(res)[...] = r.reshape(m)
                return 0
            except BaseException as e:
                errors.append(e)
                return 1

    elif problem == "gradient":
        gview = _viewer(ct, npdt, (n,))

        def cb(x, dims, cost, grad, user):
            try:
                cost[0] = fn(xin(x), None if grad is None else gview(grad))
                return 0
            except BaseException as e:
                errors.append(e)
                return 1

    else:
        gview = _viewer(ct, npdt, (n,))
        hview = _viewer(ct, npdt, (n, n), order="F")

        def cb(x, dims, cost, grad, hess, user):
            try:
                cost[0] = fn(xin(x), None if grad is None else gview(grad),
                             None if hess is None else hview(hess))
                return 0
            except BaseException as e:
                errors.append(e)
                return 1

    return cb_type(cb)


def _required_args(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 1
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(p.kind in positional and p.default is p.empty for p in params)


def _lookup(table, value, what):
    try:
        return table[value.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown {what} {value!r}, expected one of {sorted(table)}") from None


def _set_option(opts, name, value):
    ftype = _OPTION_FIELDS.get(name)
    if ftype is None or name in _RESERVED_FIELDS:
        raise TypeError(f"optimize() got an unexpected option {name!r}")
    if name == "log_error_symbol":
        value = value.encode() if isinstance(value, str) else value
    elif ftype in _INT_LIMITS:
        if not 0 <= int(value) < _INT_LIMITS[ftype]:
            raise ValueError(f"option {name!r} must be in [0, {_INT_LIMITS[ftype] - 1}]")
        value = int(value)
    elif ftype is C.c_int:
        value = int(value)
    setattr(opts, name, value)


def optimize(
    fn: Callable,
    x0,
    problem: str = "residuals",
    *,
    solver: Optional[str] = None,
    linear_solver: Optional[str] = None,
    fixed: Optional[bool] = None,
    plus_eq: Optional[Callable] = None,
    stop_callback: Optional[Callable] = None,
    **options,
) -> Result:
    """Minimize `fn` starting from `x0`.

    `x0` is a Python scalar (then `fn` receives and `Result.x` is a float) or a 1-D
    float64/float32 array (then `fn` receives a read-only zero-copy view of Tinyopt's
    parameters, valid during the call only; copy it to keep it).

    `problem` selects what `fn` provides:

    - ``"residuals"``: ``fn(x) -> r`` or ``fn(x) -> (r, J)`` with ``J`` of shape (m, n);
      without ``J``, finite differences are used. ``fn(x, jac) -> r`` (two required
      parameters) fills the Jacobian in place, like the C API: ``jac`` is a zero-copy
      writable (m, n) view, or None when Tinyopt does not need it (skip computing it).
    - ``"cost"``: ``fn(x) -> float`` (derivative-free, finite differences).
    - ``"gradient"``: ``fn(x, grad) -> cost``, accumulating into the zeroed ``grad`` (n,).
    - ``"hessian"``: ``fn(x, grad, hess) -> cost``, accumulating into the zeroed ``grad``
      (n,) and ``hess`` (n, n) views. ``grad``/``hess`` are None when not requested.

    ``solver`` is one of lm, gn, gd, cg, dogleg, bfgs, lbfgs (default lm for residuals and
    hessian, bfgs for cost and gradient). ``linear_solver`` is ldlt, llt, lu, qr, svd.
    ``fixed``: None uses the allocation-free fixed-size solver when one exists for
    ``len(x0)``, True requires it, False forces the dynamic-size one. ``plus_eq(x, dx)``
    applies a custom (e.g. manifold) update in place on array views. ``stop_callback(error,
    step_norm2, gradient_norm2)`` returns True to stop. Any other keyword sets a field of
    ``tinyopt_options_t`` (e.g. ``max_iters=100``, ``log_enabled=True``).
    """
    if problem not in _PROBLEMS:
        raise ValueError(f"unknown problem {problem!r}, expected one of {sorted(_PROBLEMS)}")
    arr = np.asarray(x0)
    if arr.ndim > 1:
        raise ValueError(f"x0 must be a scalar or a 1-D array, got shape {arr.shape}")
    scalar = arr.ndim == 0
    npdt = np.float32 if arr.dtype == np.float32 else np.float64
    x = np.array(arr, dtype=npdt).reshape(-1)  # the only copy: Tinyopt optimizes it in place
    n = x.size
    if n == 0:
        raise ValueError("x0 must not be empty")
    ct, key = (C.c_float, "f") if npdt is np.float32 else (C.c_double, "d")

    dynamic_entry = capi.entry_points.get((key, None))
    if dynamic_entry is None:
        raise RuntimeError("this Tinyopt C library was built without the float32 API")
    fixed_entry = capi.entry_points.get((key, n))
    if fixed and fixed_entry is None:
        raise ValueError(f"no fixed-size solver for {n} parameters in this library")
    entry = fixed_entry if fixed_entry is not None and fixed is not False else dynamic_entry
    use_fixed = entry is fixed_entry

    m, analytic = 1, False
    inplace_jac = problem == "residuals" and _required_args(fn) >= 2
    if problem == "residuals":  # one probing call gives the residual count and Jacobian mode
        xv = x.view()
        xv.flags.writeable = False
        xin0 = float(x[0]) if scalar else xv
        out = fn(xin0, None) if inplace_jac else fn(xin0)
        analytic = inplace_jac or type(out) is tuple
        m = np.size(out[0] if type(out) is tuple else out)
        if m == 0:
            raise ValueError("fn returned no residuals")

    opts = capi.default_options()
    opts.log_enabled = 0  # quiet by default; pass log_enabled=True to trace iterations
    if solver is None:
        solver = "bfgs" if problem in ("cost", "gradient") else "lm"
    opts.solver_type = _lookup(capi.SOLVERS, solver, "solver")
    if problem == "residuals" and solver.lower() in _FIRST_ORDER:
        raise ValueError(f"solver {solver!r} needs a scalar objective: use problem='cost' or "
                         "'gradient', or a least-squares solver (lm, gn, dogleg)")
    if linear_solver is not None:
        opts.linear_solver = _lookup(capi.LINEAR_SOLVERS, linear_solver, "linear_solver")
    for name, value in options.items():
        _set_option(opts, name, value)

    errors = []
    keep = []  # ctypes callbacks must outlive the call
    if stop_callback is not None:

        def stop(error, step2, grad2, user):
            try:
                return 1 if stop_callback(error, step2, grad2) else 0
            except BaseException as e:
                errors.append(e)
                return 1

        keep.append(capi.StopCallback(stop))
        opts.stop_callback = C.cast(keep[-1], C.c_void_p).value

    plus_ptr = None
    if plus_eq is not None:
        xw, dw = _viewer(ct, npdt, (n,)), _viewer(ct, npdt, (n,))

        def plus(xa, da):
            try:
                plus_eq(xw(xa), dw(da))
            except BaseException as e:
                errors.append(e)

        keep.append(capi.PlusEqCallback(plus))
        plus_ptr = C.cast(keep[-1], C.c_void_p).value

    cb = _build_callback(problem, fn, n, m, analytic, inplace_jac, scalar, ct, npdt, errors)
    prob = capi.Problem(_PROBLEMS[problem], C.cast(cb, C.c_void_p).value, m, None)
    params = (capi.FixedParams(x.ctypes.data, plus_ptr) if use_fixed
              else capi.Params(x.ctypes.data, n, plus_ptr))
    summary = capi.Summary()

    status = entry(C.byref(params), C.byref(prob), C.byref(opts), C.byref(summary))

    if errors:
        raise errors[0]
    if status == capi.STATUS_INVALID_ARGUMENT:
        raise ValueError(
            "Tinyopt rejected the problem: invalid argument or a solver/linear solver that "
            "is not compiled into the library")
    if status in (capi.STATUS_INTERNAL_ERROR, capi.STATUS_CALLBACK_FAILED):
        raise RuntimeError(f"Tinyopt failed with status {status}")

    return Result(
        x=float(x[0]) if scalar else x,
        cost=summary.final_cost,
        iters=summary.num_iters,
        failures=summary.num_failures,
        stop_reason=StopReason(summary.stop_reason),
        success=status == capi.STATUS_OK,
        num_residuals=summary.num_residuals,
        numerical_diff=bool(summary.used_numerical_differentiation),
    )
