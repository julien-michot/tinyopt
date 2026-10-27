// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_PROBLEM_FLOAT_H
#define TINYOPT_C_C_API_PROBLEM_FLOAT_H

#include <tinyopt/c/c_api_common.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return 0 to continue, or nonzero to stop optimization. */
typedef int (*tinyopt_cost_funcf_t)(const float *x, int dims, float *cost, void *user_data);
/* Jacobian is row-major. Leave *jacobian unchanged to fill it, or set it to NULL for numerical
  differentiation; do not replace it with another non-NULL pointer. */
typedef int (*tinyopt_residuals_funcf_t)(const float *x, int dims, float *residuals,
                                         float **jacobian, int num_residuals, void *user_data);
/* These callbacks manually accumulate the gradient and, optionally, Hessian. */
typedef int (*tinyopt_acc_grad_funcf_t)(const float *x, int dims, float *cost, float *gradient,
                                        void *user_data);
typedef int (*tinyopt_acc_hessian_funcf_t)(const float *x, int dims, float *cost, float *gradient,
                                           float *hessian, void *user_data);

typedef struct tinyopt_problemf_t {
  tinyopt_eval_type_t type; /* Callback evaluation mode. */
  union {
    tinyopt_cost_funcf_t cost;               /* Scalar objective callback. */
    tinyopt_residuals_funcf_t residuals;     /* Residual and optional Jacobian callback. */
    tinyopt_acc_grad_funcf_t acc_grad;       /* Scalar cost and gradient accumulation callback. */
    tinyopt_acc_hessian_funcf_t acc_hessian; /* Scalar cost, gradient, and Hessian callback. */
  } fn;                                      /* Callback matching type. */
  int num_residuals;                         /* Number of residuals in residual evaluation mode. */
  void *user_data; /* User data passed unchanged to the selected callback. */
} tinyopt_problemf_t;

#ifdef __cplusplus
}
#endif

#endif
