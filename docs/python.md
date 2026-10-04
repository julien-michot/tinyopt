# Python API

The `tinyopt` Python package is a thin, copy-free [ctypes](https://docs.python.org/3/library/ctypes.html)
binding of the [C API](c_api.md). It needs only `numpy` at runtime and exposes one function.

```python
import tinyopt

res = tinyopt.optimize(lambda x: x * x - 2.0, 1.0)   # sqrt(2)
print(res.x, res.converged)
```

## Install

```shell
python -m pip install .          # builds libtinyopt_c with CMake and bundles it in the wheel
```

Requirements: Python >= 3.9, NumPy, CMake >= 3.25, a C++20 compiler and Eigen (fetched
by CMake if missing). With Pixi: `pixi run pip-install`.

- The C library is built in `build/cmake-c-library` and reused by later installs, only changed
  files are rebuilt.
- To link an already built library instead (no compiler/CMake needed), set
  `TINYOPT_C_LIBRARY=/path/to/libtinyopt_c.so` when installing. The same variable, set at run
  time, overrides the library the package loads.
- `TINYOPT_CMAKE_ARGS="-DTINYOPT_C_FIXED_SIZES=2;3"` forwards extra options to CMake.

## `optimize(fn, x0, problem="residuals", **options)`

`x0` is either

- a Python scalar: `fn` receives a `float` and `Result.x` is a `float`, or
- a 1-D array (or list): `fn` receives a **read-only, zero-copy NumPy view** of Tinyopt's
  parameters. It is only valid during the call, `copy()` it to keep it.

`float32` arrays use the float C API, everything else is optimized in `float64`. `x0` is never
modified, `Result.x` is the optimized copy.

| `problem` | `fn` signature | Derivatives |
| :--- | :--- | :--- |
| `"residuals"` (default) | `fn(x) -> r` | finite differences |
| | `fn(x) -> (r, J)` | analytic, `J` is `(m, n)` |
| | `fn(x, jac) -> r` | analytic, filled **in place** (C API style) |
| `"cost"` | `fn(x) -> float` | finite differences |
| `"gradient"` | `fn(x, grad) -> cost` | accumulate into `grad` `(n,)` |
| `"hessian"` | `fn(x, grad, hess) -> cost` | accumulate into `grad` `(n,)` and `hess` `(n, n)` |

`r` may be a float or an array. In `fn(x, jac)` the Jacobian is a writable `(m, n)` view that is
`None` when Tinyopt only needs the residuals (e.g. trial steps), skip computing it then. In the
accumulation problems `grad` and `hess` are zero-copy views of Tinyopt's zeroed buffers (`None`
when not requested); write with `+=` or assignment, nothing is allocated or returned. For a
scalar `x0` they are `(1,)` and `(1, 1)` arrays. Outputs returned by `fn` (`r`, `J`) are copied
once into Tinyopt's buffers.

```python
def residuals(p, jac):                       # circle fit, analytic Jacobian in place
    d = pts - p[:2]
    dist = np.linalg.norm(d, axis=1)
    if jac is not None:
        jac[:, :2] = -d / dist[:, None]
        jac[:, 2] = -1.0
    return dist - p[2]

res = tinyopt.optimize(residuals, [0.0, 0.0, 1.0])
```

### Fixed and dynamic sizes

By default (`fixed=None`) a fixed-size, allocation-free solver is used when one exists for
`len(x0)` (sizes 1-6, 10 and 12, see `TINYOPT_C_FIXED_SIZES`), otherwise the dynamic-size one.
`fixed=True` requires the fixed-size solver (`ValueError` if unavailable), `fixed=False` forces the
dynamic one.

### Options

| Keyword | Meaning |
| :--- | :--- |
| `solver` | `"lm"` (default for residuals/hessian), `"gn"`, `"dogleg"`; `"bfgs"` (default for cost/gradient), `"lbfgs"`, `"cg"`, `"gd"` |
| `linear_solver` | `"ldlt"` (default), `"llt"`, `"lu"`, `"qr"`, `"svd"` |
| `fixed` | see above |
| `plus_eq` | `plus_eq(x, dx)` custom in-place update (manifolds), array views |
| `stop_callback` | `stop_callback(error, step_norm2, gradient_norm2) -> bool`, True stops |
| any `tinyopt_options_t` field | e.g. `max_iters=100`, `lm_damping_init=1e-3`, `log_enabled=True` |

Least-squares solvers (`lm`, `gn`, `dogleg`) are required for `"residuals"`. Logging is off by
default. Unknown keywords raise `TypeError`, wrong values `ValueError`.

### Result

`Result` has `x`, `cost`, `iters`, `failures`, `stop_reason` (`tinyopt.StopReason`), `success`
(no solver failure), `converged` (an error/step/gradient criterion was met), `num_residuals` and
`numerical_diff`.

### Errors

Exceptions raised by `fn` (including `KeyboardInterrupt`) stop the optimization and are
re-raised unchanged by `optimize()`. Wrong output sizes raise `ValueError`.

## Performance notes

- Parameters, Jacobian, gradient and Hessian are views on C memory, cached per address so
  repeated callbacks do not create new objects.
- The per-callback cost is the ctypes call plus your function; keep `fn` vectorized with NumPy.
- A residual function returning `(r, J)` always computes `J`; use `fn(x, jac)` to skip it.

## Development

```shell
pixi run test-python        # builds the C library (incrementally) and runs tests/python
pixi run examples-python    # runs examples/python
pixi run test-pip-install   # installs into a fresh virtualenv and tests the installed package
```

Examples: [examples/python](../examples/python). The ctypes `Options` mirror in
`bindings/python/tinyopt/_capi.py` must follow `tinyopt_options_t`; a test compares both and the
package refuses to import on a layout mismatch.
