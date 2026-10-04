// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_SPARSE_FLOAT_H
#define TINYOPT_C_C_API_SPARSE_FLOAT_H

#include <tinyopt/c/c_api_float.h>

#if TINYOPT_C_API_ENABLE_SUITESPARSE && TINYOPT_C_API_ENABLE_FLOAT

#include <cholmod.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Float version of tinyopt_sparse_acc_hessian_func_t; `*hessian` must be single precision
  (CHOLMOD_SINGLE), real and int-indexed. */
typedef int (*tinyopt_sparse_acc_hessian_funcf_t)(const float *x, int dims, float *cost,
                                                  float *gradient, cholmod_sparse **hessian,
                                                  cholmod_common *common, void *user_data);

typedef struct tinyopt_sparse_problemf_t {
  tinyopt_sparse_acc_hessian_funcf_t acc_hessian; /* Cost, gradient and sparse Hessian callback. */
  void *user_data;                                /* User data passed unchanged to the callback. */
} tinyopt_sparse_problemf_t;

/* Uses Levenberg-Marquardt (or Gauss-Newton) with a SuiteSparse CHOLMOD linear solver. */
TINYOPT_C_API tinyopt_status_t tinyopt_optimize_sparsef(const tinyopt_paramsf_t *params,
                                                        const tinyopt_sparse_problemf_t *problem,
                                                        const tinyopt_options_t *options,
                                                        tinyopt_summary_t *summary);

#ifdef __cplusplus
}
#endif

#endif

#endif
