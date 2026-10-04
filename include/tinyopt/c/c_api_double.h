// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_DOUBLE_H
#define TINYOPT_C_C_API_DOUBLE_H

#include <tinyopt/c/c_api_common.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tinyopt_params {
  double *x;
  int dims;
  void (*plus_eq)(double *x, double *dx);
} tinyopt_params;

/* Return 0 to continue, or nonzero to stop optimization. */
typedef int (*tinyopt_cost_func)(const double *x, int dims, double *cost, void *user_data);
/* Jacobian is row-major; the library passes NULL when use_jacobian is false. */
typedef int (*tinyopt_residuals_func)(const double *x, int dims, double *residuals,
                                      double *jacobian, int num_residuals, void *user_data);
/* These callbacks manually accumulate the gradient and, optionally, Hessian. */
typedef int (*tinyopt_acc_grad_func)(const double *x, int dims, double *cost, double *gradient,
                                     void *user_data);
typedef int (*tinyopt_acc_hessian_func)(const double *x, int dims, double *cost, double *gradient,
                                        double *hessian, void *user_data);

typedef struct tinyopt_problem {
  tinyopt_eval_type type;
  union {
    tinyopt_cost_func cost;
    tinyopt_residuals_func residuals;
    tinyopt_acc_grad_func acc_grad;
    tinyopt_acc_hessian_func acc_hessian;
  } fn;
  int num_residuals;
  int use_jacobian;
  void *user_data;
} tinyopt_problem;

/* Uses default Levenberg-Marquardt options and numerical differentiation. */
TINYOPT_C_API tinyopt_status tinyopt_optimize(const tinyopt_params *params,
                                              const tinyopt_problem *problem,
                                              const tinyopt_options *options,
                                              tinyopt_summary *summary);

#ifdef __cplusplus
}
#endif

#include <tinyopt/c/c_api_fixed_double.h>

#endif