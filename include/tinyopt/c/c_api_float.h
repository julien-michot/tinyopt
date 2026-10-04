// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_FLOAT_H
#define TINYOPT_C_C_API_FLOAT_H

#include <tinyopt/c/c_api_common.h>

#if TINYOPT_C_API_ENABLE_FLOAT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tinyopt_paramsf {
  float *x;
  int dims;
  void (*plus_eq)(float *x, float *dx);
} tinyopt_paramsf;

/* Return 0 to continue, or nonzero to stop optimization. */
typedef int (*tinyopt_cost_funcf)(const float *x, int dims, float *cost, void *user_data);
/* Jacobian is row-major; the library passes NULL when use_jacobian is false. */
typedef int (*tinyopt_residuals_funcf)(const float *x, int dims, float *residuals, float *jacobian,
                                       int num_residuals, void *user_data);
/* These callbacks manually accumulate the gradient and, optionally, Hessian. */
typedef int (*tinyopt_acc_grad_funcf)(const float *x, int dims, float *cost, float *gradient,
                                      void *user_data);
typedef int (*tinyopt_acc_hessian_funcf)(const float *x, int dims, float *cost, float *gradient,
                                         float *hessian, void *user_data);

typedef struct tinyopt_problemf {
  tinyopt_eval_type type;
  union {
    tinyopt_cost_funcf cost;
    tinyopt_residuals_funcf residuals;
    tinyopt_acc_grad_funcf acc_grad;
    tinyopt_acc_hessian_funcf acc_hessian;
  } fn;
  int num_residuals;
  int use_jacobian;
  void *user_data;
} tinyopt_problemf;

/* Uses default Levenberg-Marquardt options and numerical differentiation. */
TINYOPT_C_API tinyopt_status tinyopt_optimizef(const tinyopt_paramsf *params,
                                               const tinyopt_problemf *problem,
                                               const tinyopt_options *options,
                                               tinyopt_summary *summary);

#ifdef __cplusplus
}
#endif

#include <tinyopt/c/c_api_fixed_float.h>

#endif

#endif