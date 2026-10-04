# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""ctypes mirror of the Tinyopt C API (`tinyopt/c/c_api*.h`) and shared-library loading."""

import ctypes as C
import os
import sys
from pathlib import Path

# Keep in sync with FIXED_SIZES of cmake/GenerateFixedCAPI.cmake.
FIXED_SIZES = (1, 2, 3, 4, 5, 6, 10, 12)

STATUS_OK = 0
STATUS_INVALID_ARGUMENT = 1
STATUS_CALLBACK_FAILED = 2
STATUS_OPTIMIZATION_FAILED = 3
STATUS_INTERNAL_ERROR = 4
STATUS_USER_STOPPED = 5

EVAL_COST, EVAL_RESIDUALS, EVAL_GRADIENT, EVAL_HESSIAN = 0, 1, 2, 3

SOLVERS = {"lm": 0, "gn": 1, "gd": 2, "cg": 3, "dogleg": 4, "bfgs": 5, "lbfgs": 6}
LINEAR_SOLVERS = {"ldlt": 0, "llt": 1, "lu": 2, "qr": 3, "svd": 4, "suitesparse": 5}


class Summary(C.Structure):
    _fields_ = [
        ("stop_reason", C.c_int),
        ("num_iters", C.c_int),
        ("num_failures", C.c_int),
        ("num_residuals", C.c_int),
        ("final_cost", C.c_double),
        ("used_numerical_differentiation", C.c_int),
    ]


_i, _f, _d = C.c_int, C.c_float, C.c_double


class Options(C.Structure):
    """Mirror of `tinyopt_options_t`; field order must match c_api_common.h."""

    _fields_ = (
        [("solver_type", _i), ("linear_solver", _i), ("svd_relative_threshold", _d)]
        + [("check_final_cost", _i), ("use_step_quality_approx", _i), ("grad_clipping", _f)]
        + [("hessian_is_full", _i), ("check_min_hessian_diagonal", _f), ("save_last_hessian", _i)]
        + [("use_squared_norm", _i), ("downscale_cost_by_two", _i), ("normalize_cost", _i)]
        + [("max_iters", C.c_ushort), ("min_error", _f), ("min_relative_error_decrease", _f)]
        + [("min_step_norm_squared", _f), ("min_gradient_norm_squared", _f)]
        + [("max_total_failures", C.c_ubyte), ("max_consecutive_failures", C.c_ubyte)]
        + [("max_duration_ms", _d)]
        + [("stop_callback", C.c_void_p), ("stop_callback_user_data", C.c_void_p)]
        + [("stop_callback2", C.c_void_p), ("stop_callback2_user_data", C.c_void_p)]
        + [("log_enabled", _i), ("log_error_symbol", C.c_char_p)]
        + [(f"log_print_{n}", _i) for n in (
            "emoji", "x", "dx", "inliers", "time", "jacobian_jet",
            "max_standard_deviation", "failure")]
        + [("lm_jacobi_scaling", _i)]
        + [(f"lm_{n}", _f) for n in (
            "damping_init", "damping_min", "damping_max", "good_factor", "bad_factor")]
        + [("gd_learning_rate", _f), ("cg_step_size", _f), ("cg_step_reduction", _f)]
        + [(f"dogleg_{n}", _f) for n in (
            "radius_init", "radius_max", "shrink_factor", "expand_factor")]
        + [(f"{s}_{n}", _f) for s in ("bfgs", "lbfgs") for n in (
            "step_size", "step_reduction", "step_growth", "max_step_size", "curvature_threshold")]
        + [("lbfgs_history_size", C.c_ubyte)]
    )


# Float and double descriptors only differ by pointee types, so one layout covers both.
class Params(C.Structure):  # tinyopt_params[f]_t
    _fields_ = [("x", C.c_void_p), ("dims", C.c_int), ("plus_eq", C.c_void_p)]


class FixedParams(C.Structure):  # tinyopt_params<N>[f]_t
    _fields_ = [("x", C.c_void_p), ("plus_eq", C.c_void_p)]


class Problem(C.Structure):  # tinyopt_problem[f]_t; `fn` is the callback union
    _fields_ = [
        ("type", C.c_int),
        ("fn", C.c_void_p),
        ("num_residuals", C.c_int),
        ("user_data", C.c_void_p),
    ]


StopCallback = C.CFUNCTYPE(C.c_int, C.c_double, C.c_double, C.c_double, C.c_void_p)
PlusEqCallback = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)


def callback_types(ct):
    """CFUNCTYPEs of the cost/residual/gradient/Hessian callbacks for scalar ctype `ct`."""
    p, v, i = C.c_void_p, C.c_void_p, C.c_int
    return {
        EVAL_COST: C.CFUNCTYPE(i, p, i, C.POINTER(ct), v),
        EVAL_RESIDUALS: C.CFUNCTYPE(i, p, i, p, C.POINTER(C.c_void_p), i, v),
        EVAL_GRADIENT: C.CFUNCTYPE(i, p, i, C.POINTER(ct), p, v),
        EVAL_HESSIAN: C.CFUNCTYPE(i, p, i, C.POINTER(ct), p, p, v),
    }


def _library_names():
    if sys.platform == "win32":
        return ["tinyopt_c.dll", "libtinyopt_c.dll"]
    if sys.platform == "darwin":
        return ["libtinyopt_c.dylib"]
    return ["libtinyopt_c.so"]


def _candidates():
    env = os.environ.get("TINYOPT_C_LIBRARY")
    if env:
        yield Path(env)
    here = Path(__file__).resolve().parent
    for name in _library_names():
        yield here / name
    prefixes = [Path(sys.prefix), Path("/usr/local"), Path("/usr")]
    for prefix in prefixes:
        for name in _library_names():
            for sub in ("lib", "lib64", "bin"):
                yield prefix / sub / name
            for sub in ("lib/x86_64-linux-gnu", "lib/aarch64-linux-gnu"):
                yield prefix / sub / name


def _load():
    tried = []
    for path in _candidates():
        if not path.is_file():
            continue
        try:
            return C.CDLL(str(path)), path
        except OSError as e:  # present but unloadable (missing dependency, wrong arch...)
            tried.append(f"{path}: {e}")
    hint = "\n  ".join(tried) if tried else "no libtinyopt_c found"
    raise ImportError(
        "Cannot load the Tinyopt C library (libtinyopt_c). Reinstall the package with "
        "`pip install .` or set TINYOPT_C_LIBRARY to its path.\n  " + hint
    )


lib, library_path = _load()


def _bind(name, params_type):
    fn = getattr(lib, name, None)
    if fn is not None:
        fn.restype = C.c_int
        fn.argtypes = [
            C.POINTER(params_type), C.POINTER(Problem),
            C.POINTER(Options), C.POINTER(Summary)]
    return fn


# {(dtype char, size or None): function}; size None is the dynamic-size entry point.
entry_points = {}
for _c, _suffix in (("d", ""), ("f", "f")):
    _fn = _bind(f"tinyopt_optimize{_suffix}", Params)
    if _fn is not None:
        entry_points[(_c, None)] = _fn
    for _n in FIXED_SIZES:
        _fn = _bind(f"tinyopt_optimize{_n}{_suffix}", FixedParams)
        if _fn is not None:
            entry_points[(_c, _n)] = _fn

if ("d", None) not in entry_points:
    raise ImportError(f"{library_path} does not export tinyopt_optimize: unsupported library")

_options_default = lib.tinyopt_options_default
_options_default.restype = C.c_int
_options_default.argtypes = [C.POINTER(Options)]


def default_options():
    opts = Options()
    if _options_default(C.byref(opts)) != STATUS_OK:
        raise RuntimeError("tinyopt_options_default failed")
    return opts


def _check_options_layout():
    """Detect a Python/C `tinyopt_options_t` layout mismatch before it corrupts memory."""
    size = C.sizeof(Options)
    pad = 256
    written = 0
    for fill in (0xA5, 0x5A):
        buf = (C.c_char * (size + pad))()
        C.memset(buf, fill, size + pad)
        _options_default(C.cast(buf, C.POINTER(Options)))
        raw = bytes(buf)
        end = max((i + 1 for i, b in enumerate(raw) if b != fill), default=0)
        written = max(written, end)
    # Round up for tail padding (struct alignment is 8).
    if (written + 7) // 8 * 8 != size:
        raise ImportError(
            f"tinyopt_options_t layout mismatch between {library_path} ({written} bytes) "
            f"and the Python binding ({size} bytes): the library and package versions differ."
        )


_check_options_layout()
