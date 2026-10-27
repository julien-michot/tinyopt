// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_BLOCK_TRIDIAG_DOUBLE_H
#define TINYOPT_C_C_API_BLOCK_TRIDIAG_DOUBLE_H

#include <tinyopt/c/c_api_common.h>

#if TINYOPT_C_API_ENABLE_DYNAMIC

#include <tinyopt/c/c_api_double.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed handle, valid only while an accumulation callback runs. */
typedef struct tinyopt_block_tridiag_t tinyopt_block_tridiag_t;

TINYOPT_C_API int tinyopt_block_tridiag_dims(const tinyopt_block_tridiag_t *hessian);
TINYOPT_C_API int tinyopt_block_tridiag_block_size(const tinyopt_block_tridiag_t *hessian);
TINYOPT_C_API int tinyopt_block_tridiag_num_blocks(const tinyopt_block_tridiag_t *hessian);

/* Writes a column-major B x B block. Returns 0 on success, nonzero for an invalid coordinate.
   Rows outside the diagonal and super-diagonal are not stored; lower-triangular writes are mirrored
   into the stored upper triangle. */
TINYOPT_C_API int tinyopt_block_tridiag_set_block(tinyopt_block_tridiag_t *hessian, int row,
                                                  int col, const double *data);

typedef int (*tinyopt_block_tridiag_acc_hessian_func_t)(const double *x, int dims, double *cost,
                                                        double *gradient,
                                                        tinyopt_block_tridiag_t *hessian,
                                                        void *user_data);

typedef struct tinyopt_block_tridiag_problem_t {
  int num_blocks; /* Number of diagonal blocks; dims must equal num_blocks * block_size. */
  int block_size; /* Square block dimension. */
  tinyopt_block_tridiag_acc_hessian_func_t acc_hessian; /* Cost, gradient and Hessian callback. */
  void *user_data; /* User data passed unchanged to the callback. */
} tinyopt_block_tridiag_problem_t;

/* Uses Levenberg-Marquardt by default; Gauss-Newton and Dogleg are also accepted. */
TINYOPT_C_API tinyopt_status_t tinyopt_optimize_block_tridiag(
    const tinyopt_params_t *params, const tinyopt_block_tridiag_problem_t *problem,
    const tinyopt_options_t *options, tinyopt_summary_t *summary);

#ifdef __cplusplus
}
#endif

#endif /* TINYOPT_C_API_ENABLE_DYNAMIC */

#endif
