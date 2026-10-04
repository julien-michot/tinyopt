# C API

Tinyopt provides a C ABI in the optional `tinyopt_c` library. Enable it with
`-DTINYOPT_BUILD_C_LIBRARY=ON`; it is off by default. Include
`<tinyopt/c/c_api.h>` for the enabled APIs, or include only
`<tinyopt/c/c_api_float.h>` / `<tinyopt/c/c_api_double.h>` when a consumer needs one precision.
The float entrypoints are built by default; configure with `-DTINYOPT_C_API_FLOAT=OFF` to omit them.
For install and CMake consumer examples, see [Installation and Usage](installation_and_usage.md).

Callbacks return `0` to continue. A nonzero result stops optimization with
`TINYOPT_STATUS_USER_STOPPED`. User data and input parameter arrays remain caller-owned and must stay
valid until return. Tinyopt optimizes a private parameter copy and copies it back on success or
requested stop.

Select a `tinyopt_problem_t` mode:

| Mode | Callback | Behavior |
| --- | --- | --- |
| `TINYOPT_EVAL_COST_ONLY` | `fn.cost` | Scalar objective; Tinyopt estimates its gradient numerically. With default LM options, C cost-only mode selects BFGS. |
| `TINYOPT_EVAL_RESIDUALS` | `fn.residuals` | Residuals with an optional row-major Jacobian. Set `*jacobian = NULL` in the callback to request numerical differentiation. |
| `TINYOPT_EVAL_GRADIENT` | `fn.acc_grad` | Manually accumulate a scalar objective and gradient; select a first-order solver. |
| `TINYOPT_EVAL_HESSIAN` | `fn.acc_hessian` | Manually accumulate a scalar objective, gradient, and Hessian. |

Tinyopt zeros gradient and Hessian buffers before accumulation callbacks. Either output pointer can be
`NULL` when the selected solver does not request it. `acc_grad` and `acc_hessian` are the accumulation
callback types; there are no separate `acc1` or `acc2` modes.

## Dynamic Parameters

Dynamic parameters include their dimension and may provide a `plus_eq` callback to apply a
parameter-space step. Set `plus_eq` to `NULL` for ordinary component-wise addition; provide a custom
callback for manifold or otherwise non-Euclidean updates. Residual callbacks fill every residual
entry on each invocation. The `jacobian` argument is a
pointer to Tinyopt's row-major Jacobian buffer. Fill it when `*jacobian` is non-`NULL`; set
`*jacobian = NULL` to ask Tinyopt to estimate the Jacobian with finite differences. The callback may
also receive a null `*jacobian` when only residuals are being evaluated, so check it before writing.

```c
#include <stdio.h>
#include <tinyopt/c/c_api.h>

static void plus_eq(double *x, double *dx) {
  x[0] += dx[0];
}

static int evaluate_residuals(const double *x, int dims, double *out, double **jacobian,
                              int residual_dims, void *user_data) {
  const double target = *(const double *)user_data;
  if (dims != 1 || residual_dims != 1) return 1;
  out[0] = x[0] - target;
  if (*jacobian != NULL) (*jacobian)[0] = 1.0;
  return 0;
}

int main(void) {
  double x[] = {-3.0};
  double target = 0.75;
  tinyopt_params_t params = {x, 1, plus_eq};
  tinyopt_problem_t problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = evaluate_residuals;
  problem.num_residuals = 1;
  problem.user_data = &target;
  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  tinyopt_summary_t summary;

  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK)
    return 1;
  printf("x = %.6f, cost = %.3g\n", x[0], summary.final_cost);
  return 0;
}
```

For a scalar objective, set `type` to `TINYOPT_EVAL_COST_ONLY` and provide `fn.cost`. The callback
receives a `double *cost` output. With default LM options the C API selects BFGS for this scalar
objective mode; an explicitly selected first-order solver is also respected.

For manually supplied derivatives, use `TINYOPT_EVAL_GRADIENT` with `fn.acc_grad` or
`TINYOPT_EVAL_HESSIAN` with `fn.acc_hessian`. These callbacks accumulate the scalar objective's
gradient and, for Hessian mode, its Hessian. The gradient buffer is a vector; the Hessian buffer is
column-major. Tinyopt initializes requested buffers to zero before each callback, so write
contributions with `+=` or assign the complete result.

Pass `NULL` as the options pointer to use Tinyopt's default C++ `Options` values.
The options initializer returns those defaults, after which any field can be overridden:

```c
tinyopt_options_t options;
tinyopt_options_default(&options);
options.max_iters = 100;
options.lm_damping_init = 1e-3f;
options.log_enabled = 0;
```

`tinyopt_options_t` mirrors the value fields in `tinyopt::Options`: solver and linear solver selection,
optimization and Hessian settings, cost scaling, stop criteria, logging, and all solver-specific
settings. The two stop callbacks are also available as C function pointers. The scalar callback
receives the current error, squared step norm, and squared gradient norm. Each callback receives its
own user-data pointer. The vector callback receives float `x` and `dx` arrays with an explicit
dimension, matching Tinyopt's existing float callback contract.

`step_callback` (`tinyopt_step_callback_t`) reports every parameter update, which makes it possible
to record the optimizer path. It receives the float step `dx` added to the parameters, its
dimension, and `is_rollback`, nonzero when a rejected step is undone (`dx` is then the negated
step). Summing the `dx` of all calls tracks the current parameters. Return nonzero to stop with
`TINYOPT_STATUS_USER_STOPPED`.

```c
static int on_step(const float *dx, int dims, int is_rollback, void *user_data) {
  double *x = (double *)user_data;  /* starts at the initial parameters */
  for (int i = 0; i < dims; ++i) x[i] += dx[i];  /* a rollback already carries the negated step */
  (void)is_rollback;
  return 0;
}
/* options.step_callback = on_step; options.step_callback_user_data = x_tracked; */
```

## Sparse Hessians (SuiteSparse)

When Tinyopt is configured with `-DTINYOPT_ENABLE_SUITESPARSE=ON` together with the C library,
`<tinyopt/c/c_api_sparse.h>` (or `c_api_sparse_double.h` / `c_api_sparse_float.h`) exposes
`tinyopt_optimize_sparse()` and `tinyopt_optimize_sparsef()`. Parameters are the usual dynamic
`tinyopt_params_t` / `tinyopt_paramsf_t`; the accumulation callback provides the cost, the gradient
and a dynamic-size sparse Hessian as a SuiteSparse `cholmod_sparse` matrix. Systems are solved with
CHOLMOD, using Levenberg-Marquardt by default or Gauss-Newton when selected in the options.

The callback receives a `cholmod_common *` to allocate with. When `hessian` is not `NULL`, set
`*hessian` to a newly allocated `dims x dims`, real, int-indexed `cholmod_sparse` of the matching
precision (`CHOLMOD_DOUBLE` or `CHOLMOD_SINGLE`); Tinyopt takes ownership and frees it. `stype` may be
`1` (upper triangle), `-1` (lower triangle) or `0` (symmetric, upper triangle is read). The gradient
buffer is zeroed beforehand; `hessian` and `gradient` are `NULL` when only the cost is requested.

```c
static int accumulate(const double *x, int dims, double *cost, double *gradient,
                      cholmod_sparse **hessian, cholmod_common *common, void *user_data) {
  *cost = 0.5 * x[0] * x[0];
  if (gradient != NULL) gradient[0] = x[0];
  if (hessian != NULL) {
    cholmod_triplet *t = cholmod_allocate_triplet(1, 1, 1, 1, CHOLMOD_REAL + CHOLMOD_DOUBLE, common);
    ((int *)t->i)[0] = ((int *)t->j)[0] = 0;
    ((double *)t->x)[0] = 1.0;
    t->nnz = 1;
    *hessian = cholmod_triplet_to_sparse(t, 1, common);
    cholmod_free_triplet(&t, common);
  }
  return 0;
}

double x[] = {3.0};
tinyopt_params_t params = {x, 1, NULL};
tinyopt_sparse_problem_t problem = {accumulate, NULL};
tinyopt_optimize_sparse(&params, &problem, NULL, NULL);
```

Only the Levenberg-Marquardt and Gauss-Newton solvers are supported; other solvers return
`TINYOPT_STATUS_INVALID_ARGUMENT`. See `examples/c/sparse_hessian.c` and `tests/c/test_api_sparse.c`.

## Fixed-Size Parameters

CMake generates fixed-size declarations, one C wrapper source per dimension and precision, and tests
for every configured combination. The default dimensions are `1, 2, 3, 4, 5, 6, 10, 12`; configure
a subset with `-DTINYOPT_C_FIXED_SIZES="2;3;6"`. Fixed-size parameter structs omit `dims`, and
their optional `plus_eq` member has the same component-wise fallback when set to `NULL`:

```c
#include <tinyopt/c/c_api.h>

static void plus_eq(float *x, float *dx) {
  for (int i = 0; i < 3; ++i) x[i] += dx[i];
}

static int evaluate_residuals(const float *x, int dims, float *out, float **jacobian,
                              int residual_dims, void *user_data) {
  const float *target = (const float *)user_data;
  if (dims != 3 || residual_dims != 3) return 1;
  for (int i = 0; i < 3; ++i) {
    out[i] = x[i] - target[i];
    if (*jacobian != NULL) {
      for (int j = 0; j < 3; ++j) (*jacobian)[i * 3 + j] = i == j ? 1.0f : 0.0f;
    }
  }
  return 0;
}

int main(void) {
  float x[3] = {2.0f, 3.0f, 4.0f};
  const float target[3] = {1.0f, 1.0f, 1.0f};
  tinyopt_params3f_t params = {x, plus_eq};
  tinyopt_problemf_t problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = evaluate_residuals;
  problem.num_residuals = 3;
  problem.user_data = (void *)target;
  return tinyopt_optimize3f(&params, &problem, NULL, NULL) != TINYOPT_STATUS_OK;
}
```

All C typedef names end in `_t`. Double is the default and has no precision suffix, for example
`tinyopt_params_t` and `tinyopt_optimize`, or `tinyopt_params3_t` and `tinyopt_optimize3` for fixed
size. Float names keep their `f` precision suffix, for example `tinyopt_paramsf_t` and
`tinyopt_optimizef`, or `tinyopt_params3f_t` and `tinyopt_optimize3f` for fixed size. Descriptors
are passed by pointer. Options and summary may be
`NULL`; parameter and residual descriptors must not be. The corresponding generated declarations
are available from the umbrella header or the precision-only header. Each generated C wrapper is a
separate translation unit so parallel build tools can compile them independently.

## WebAssembly

The C API can be compiled to WebAssembly with Emscripten and called from JavaScript, see
[examples/wasm](../examples/wasm/README.md). In the `wasm` pixi environment, `pixi run build-wasm`
cross-compiles `tinyopt_c` (static, double precision, 2D fixed size) in `build-wasm/` with
`-DTINYOPT_BUILD_WASM=ON` and links it into `tinyopt.mjs` / `tinyopt.wasm`, exporting
`tinyopt_optimize`, `tinyopt_options_default`, `malloc` and `free`. JavaScript fills the C structs
in the module memory (wasm32 layout) and passes functions as callbacks with `addFunction`.
C++ exceptions are enabled (`-fexceptions`) because the C API reports callback stops with them.

```js
const m = await createTinyopt();
// ... write tinyopt_params_t / tinyopt_problem_t / tinyopt_options_t, set
// options.step_callback = m.addFunction((dx, dims, isRollback, user) => 0, 'iiiii');
const status = m._tinyopt_optimize(params, problem, options, summary);
```

## Current Boundaries

The C ABI does not expose C++ templated autodiff. Fixed-size entrypoints compile with
`TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS` and Eigen's runtime no-malloc guard; C API adapter scratch is
allocated before optimization begins. The guard cannot prevent allocations made by user callbacks.
Dynamic-size solver/workspace storage is runtime-sized, so dynamic parameters cannot promise zero
allocation. A fixed dimension must be present in `TINYOPT_C_FIXED_SIZES`; dynamic entrypoints accept
any positive dimension. C++ exceptions must not cross the C callback boundary.
