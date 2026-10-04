# C API

Tinyopt provides a C ABI in the `tinyopt_c` shared library. Include
`<tinyopt/c/c_api.h>` for the enabled APIs, or include only
`<tinyopt/c/c_api_float.h>` / `<tinyopt/c/c_api_double.h>` when a consumer needs one precision.
The float entrypoints are built by default; configure with `-DTINYOPT_C_API_FLOAT=OFF` to omit them.

C callbacks return `0` on success. A nonzero residual callback result stops optimization and returns
`TINYOPT_STATUS_RESIDUAL_CALLBACK_FAILED`. Callback `user_data` and parameter arrays remain owned by
the caller and must remain valid until the call returns. Tinyopt works on a private parameter copy
and copies the optimized values back only after success. The API uses numerical differentiation.

## Dynamic Parameters

Dynamic parameters include their dimension and a `plus_eq` callback, which applies a parameter-space
step. Residual callbacks fill every entry of their fixed-size residual output on each invocation.

```c
#include <stdio.h>
#include <tinyopt/c/c_api.h>

static void plus_eq(double *x, double *dx) {
  x[0] += dx[0];
}

static int evaluate_residuals(const double *x, int dims, double *out, int residual_dims,
                              void *user_data) {
  const double target = *(const double *)user_data;
  if (dims != 1 || residual_dims != 1) return 1;
  out[0] = x[0] - target;
  return 0;
}

int main(void) {
  double x[] = {-3.0};
  double target = 0.75;
  tinyopt_params params = {x, 1, plus_eq};
  tinyopt_residuals residual_func = {evaluate_residuals, 1, &target};
  tinyopt_options options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  tinyopt_summary summary;

  if (tinyopt_optimize(&params, &residual_func, &options, &summary) != TINYOPT_STATUS_OK)
    return 1;
  printf("x = %.6f, cost = %.3g\n", x[0], summary.final_cost);
  return 0;
}
```

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

static int evaluate_residuals(const float *x, int dims, float *out, int residual_dims,
                              void *user_data) {
  const float *target = (const float *)user_data;
  if (dims != 3 || residual_dims != 3) return 1;
  for (int i = 0; i < 3; ++i) out[i] = x[i] - target[i];
  return 0;
}

int main(void) {
  float x[3] = {2.0f, 3.0f, 4.0f};
  const float target[3] = {1.0f, 1.0f, 1.0f};
  tinyopt_params3f params = {x, plus_eq};
  tinyopt_residualsf residual_func = {evaluate_residuals, 3, (void *)target};
  return tinyopt_optimize3f(&params, &residual_func, NULL, NULL) != TINYOPT_STATUS_OK;
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

The C ABI currently accepts residual callbacks and computes derivatives numerically; it does not
expose Tinyopt's C++ templated autodiff or direct gradient/Hessian accumulation interfaces. C++
exception propagation across a C callback boundary is unsupported; callbacks should report failure
through their integer return value. A dimension must be present in `TINYOPT_C_FIXED_SIZES` to use its
fixed-size symbol, while dynamic-size entrypoints support any positive dimension at runtime.
