# C API

Tinyopt provides a C ABI in the `tinyopt_c` shared library. Include
`<tinyopt/c/c_api.h>` for the enabled APIs, or include only
`<tinyopt/c/c_api_float.h>` / `<tinyopt/c/c_api_double.h>` when a consumer needs one precision.
The float entrypoints are built by default; configure with `-DTINYOPT_C_API_FLOAT=OFF` to omit them.

Callbacks return `0` to continue. A nonzero result stops optimization with
`TINYOPT_STATUS_USER_STOPPED`. User data and input parameter arrays remain caller-owned and must stay
valid until return. Tinyopt optimizes a private parameter copy and copies it back on success or
requested stop.

Select a `tinyopt_problem` mode:

| Mode | Callback | Behavior |
| --- | --- | --- |
| `TINYOPT_EVAL_COST_ONLY` | `fn.cost` | Scalar objective; Tinyopt estimates its gradient numerically. With default LM options, C cost-only mode selects BFGS. |
| `TINYOPT_EVAL_RESIDUALS` | `fn.residuals` | Residuals with optional row-major Jacobian. Set `use_jacobian = 0` for a numeric Jacobian. |
| `TINYOPT_EVAL_GRADIENT` | `fn.acc_grad` | Manually accumulate a scalar objective and gradient; select a first-order solver. |
| `TINYOPT_EVAL_HESSIAN` | `fn.acc_hessian` | Manually accumulate a scalar objective, gradient, and Hessian. |

Tinyopt zeros gradient and Hessian buffers before accumulation callbacks. Either output pointer can be
`NULL` when the selected solver does not request it. `acc_grad` and `acc_hessian` are the accumulation
callback types; there are no separate `acc1` or `acc2` modes.

## Dynamic Parameters

Dynamic parameters include their dimension and a `plus_eq` callback, which applies a parameter-space
step. Residual callbacks fill every residual entry on each invocation. If `use_jacobian` is nonzero,
they also fill a row-major Jacobian; otherwise Tinyopt passes `NULL` for the Jacobian and estimates it
with finite differences.

```c
#include <stdio.h>
#include <tinyopt/c/c_api.h>

static void plus_eq(double *x, double *dx) {
  x[0] += dx[0];
}

static int evaluate_residuals(const double *x, int dims, double *out, double *jacobian,
                              int residual_dims, void *user_data) {
  const double target = *(const double *)user_data;
  if (dims != 1 || residual_dims != 1) return 1;
  out[0] = x[0] - target;
  if (jacobian != NULL) jacobian[0] = 1.0;
  return 0;
}

int main(void) {
  double x[] = {-3.0};
  double target = 0.75;
  tinyopt_params params = {x, 1, plus_eq};
  tinyopt_problem problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = evaluate_residuals;
  problem.num_residuals = 1;
  problem.use_jacobian = 1;
  problem.user_data = &target;
  tinyopt_options options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  tinyopt_summary summary;

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
tinyopt_options options;
tinyopt_options_default(&options);
options.max_iters = 100;
options.lm_damping_init = 1e-3f;
options.log_enabled = 0;
```

`tinyopt_options` mirrors the value fields in `tinyopt::Options`: solver and linear solver selection,
optimization and Hessian settings, cost scaling, stop criteria, logging, and all solver-specific
settings. The two stop callbacks are also available as C function pointers. The scalar callback
receives the current error, squared step norm, and squared gradient norm. Each callback receives its
own user-data pointer. The vector callback receives float `x` and `dx` arrays with an explicit
dimension, matching Tinyopt's existing float callback contract.

## Fixed-Size Parameters

CMake generates fixed-size declarations, one C wrapper source per dimension and precision, and tests
for every configured combination. The default dimensions are `1, 2, 3, 4, 5, 6, 10, 12`; configure
a subset with `-DTINYOPT_C_FIXED_SIZES="2;3;6"`. Fixed-size parameter structs omit `dims`:

```c
#include <tinyopt/c/c_api.h>

static void plus_eq(float *x, float *dx) {
  for (int i = 0; i < 3; ++i) x[i] += dx[i];
}

static int evaluate_residuals(const float *x, int dims, float *out, float *jacobian,
                              int residual_dims, void *user_data) {
  const float *target = (const float *)user_data;
  if (dims != 3 || residual_dims != 3) return 1;
  for (int i = 0; i < 3; ++i) {
    out[i] = x[i] - target[i];
    if (jacobian != NULL) {
      for (int j = 0; j < 3; ++j) jacobian[i * 3 + j] = i == j ? 1.0f : 0.0f;
    }
  }
  return 0;
}

int main(void) {
  float x[3] = {2.0f, 3.0f, 4.0f};
  const float target[3] = {1.0f, 1.0f, 1.0f};
  tinyopt_params3f params = {x, plus_eq};
  tinyopt_problemf problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = evaluate_residuals;
  problem.num_residuals = 3;
  problem.use_jacobian = 1;
  problem.user_data = (void *)target;
  return tinyopt_optimize3f(&params, &problem, NULL, NULL) != TINYOPT_STATUS_OK;
}
```

Double is the default and has no precision suffix, for example `tinyopt_params` and
`tinyopt_optimize`, or `tinyopt_params3` and `tinyopt_optimize3` for fixed size. Float names end in
`f`, for example `tinyopt_paramsf` and `tinyopt_optimizef`, or `tinyopt_params3f` and
`tinyopt_optimize3f` for fixed size. Descriptors are passed by pointer. Options and summary may be
`NULL`; parameter and residual descriptors must not be. The corresponding generated declarations
are available from the umbrella header or the precision-only header. Each generated C wrapper is a
separate translation unit so parallel build tools can compile them independently.

## Current Boundaries

The C ABI does not expose C++ templated autodiff. Fixed-size entrypoints compile with
`TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS` and Eigen's runtime no-malloc guard; C API adapter scratch is
allocated before optimization begins. The guard cannot prevent allocations made by user callbacks.
Dynamic-size solver/workspace storage is runtime-sized, so dynamic parameters cannot promise zero
allocation. A fixed dimension must be present in `TINYOPT_C_FIXED_SIZES`; dynamic entrypoints accept
any positive dimension. C++ exceptions must not cross the C callback boundary.
