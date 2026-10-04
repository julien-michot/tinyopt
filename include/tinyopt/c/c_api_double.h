// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_DOUBLE_H
#define TINYOPT_C_C_API_DOUBLE_H

#include <tinyopt/c/c_api_common.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tinyopt_params_t {
  double *x; /* In/out parameter values; updated on success or requested stop. */
  int dims;  /* Number of parameter values; must be positive. */
  void (*plus_eq)(double *x, double *dx); /* Optional update; NULL uses component-wise addition. */
} tinyopt_params_t;

/* Return 0 to continue, or nonzero to stop optimization. */
typedef int (*tinyopt_cost_func_t)(const double *x, int dims, double *cost, void *user_data);
/* Jacobian is row-major. Leave *jacobian unchanged to fill it, or set it to NULL for numerical
  differentiation; do not replace it with another non-NULL pointer. */
typedef int (*tinyopt_residuals_func_t)(const double *x, int dims, double *residuals,
                                        double **jacobian, int num_residuals, void *user_data);
/* These callbacks manually accumulate the gradient and, optionally, Hessian. */
typedef int (*tinyopt_acc_grad_func_t)(const double *x, int dims, double *cost, double *gradient,
                                       void *user_data);
typedef int (*tinyopt_acc_hessian_func_t)(const double *x, int dims, double *cost, double *gradient,
                                          double *hessian, void *user_data);

typedef struct tinyopt_problem_t {
  tinyopt_eval_type_t type; /* Callback evaluation mode. */
  union {
    tinyopt_cost_func_t cost;               /* Scalar objective callback. */
    tinyopt_residuals_func_t residuals;     /* Residual and optional Jacobian callback. */
    tinyopt_acc_grad_func_t acc_grad;       /* Scalar cost and gradient accumulation callback. */
    tinyopt_acc_hessian_func_t acc_hessian; /* Scalar cost, gradient, and Hessian callback. */
  } fn;                                     /* Callback matching type. */
  int num_residuals;                        /* Number of residuals in residual evaluation mode. */
  void *user_data; /* User data passed unchanged to the selected callback. */
} tinyopt_problem_t;

/* Uses default Levenberg-Marquardt options and numerical differentiation. */
TINYOPT_C_API tinyopt_status_t tinyopt_optimize(const tinyopt_params_t *params,
                                                const tinyopt_problem_t *problem,
                                                const tinyopt_options_t *options,
                                                tinyopt_summary_t *summary);

#ifdef __cplusplus
}
#endif

#include <tinyopt/c/c_api_fixed_double.h>

#endif