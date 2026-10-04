// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_SPARSE_DOUBLE_H
#define TINYOPT_C_C_API_SPARSE_DOUBLE_H

#include <tinyopt/c/c_api_double.h>

#if TINYOPT_C_API_ENABLE_SUITESPARSE

#include <cholmod.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Accumulates the scalar cost, gradient and sparse Hessian (dims x dims, dynamic size).
  Return 0 to continue, or nonzero to stop optimization.
  `gradient` and `hessian` are NULL when only the cost is requested. `gradient` is zeroed
  beforehand. When `hessian` is not NULL, set `*hessian` to a newly allocated double precision,
  real, int-indexed cholmod_sparse (e.g. with cholmod_triplet_to_sparse()) using `common`; Tinyopt
  takes ownership and frees it. Its stype may be 0 (symmetric, upper triangle is read), 1 (upper) or
  -1 (lower). */
typedef int (*tinyopt_sparse_acc_hessian_func_t)(const double *x, int dims, double *cost,
                                                 double *gradient, cholmod_sparse **hessian,
                                                 cholmod_common *common, void *user_data);

typedef struct tinyopt_sparse_problem_t {
  tinyopt_sparse_acc_hessian_func_t acc_hessian; /* Cost, gradient and sparse Hessian callback. */
  void *user_data;                               /* User data passed unchanged to the callback. */
} tinyopt_sparse_problem_t;

/* Uses Levenberg-Marquardt (or Gauss-Newton) with a SuiteSparse CHOLMOD linear solver. */
TINYOPT_C_API tinyopt_status_t tinyopt_optimize_sparse(const tinyopt_params_t *params,
                                                       const tinyopt_sparse_problem_t *problem,
                                                       const tinyopt_options_t *options,
                                                       tinyopt_summary_t *summary);

#ifdef __cplusplus
}
#endif

#endif

#endif
