// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_BLOCK_SPARSE_DOUBLE_H
#define TINYOPT_C_C_API_BLOCK_SPARSE_DOUBLE_H

#include <tinyopt/c/c_api_common.h>

#if TINYOPT_C_API_ENABLE_DYNAMIC

#include <tinyopt/c/c_api_double.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed handle, valid only while an accumulation callback runs. */
typedef struct tinyopt_block_sparse_t tinyopt_block_sparse_t;

/* Runtime block layout: one size and one count per block type. Block indices are global: types are
   laid out consecutively in the order given here. */
typedef struct tinyopt_block_sparse_layout_t {
  const int *block_sizes;  /* Positive block sizes, one per block type. */
  const int *block_counts; /* Number of blocks of each type. */
  int num_block_types;     /* Number of block types; must be positive. */
} tinyopt_block_sparse_layout_t;

TINYOPT_C_API int tinyopt_block_sparse_dims(const tinyopt_block_sparse_t *hessian);
TINYOPT_C_API int tinyopt_block_sparse_num_block_types(const tinyopt_block_sparse_t *hessian);
TINYOPT_C_API int tinyopt_block_sparse_block_size(const tinyopt_block_sparse_t *hessian, int type);
TINYOPT_C_API int tinyopt_block_sparse_num_blocks(const tinyopt_block_sparse_t *hessian, int type);
/* Scalar offset of the first block of the given type. */
TINYOPT_C_API int tinyopt_block_sparse_block_offset(const tinyopt_block_sparse_t *hessian,
                                                    int type);
/* Dimensions of the block at global block coordinates (row, col). Returns 0 on success. */
TINYOPT_C_API int tinyopt_block_sparse_block_dims(const tinyopt_block_sparse_t *hessian, int row,
                                                  int col, int *rows, int *cols);
/* Writes a column-major block at global block coordinates. Returns 0 on success, nonzero for an
   invalid coordinate. Lower-triangular writes are mirrored into the stored upper triangle. */
TINYOPT_C_API int tinyopt_block_sparse_set_block(tinyopt_block_sparse_t *hessian, int row, int col,
                                                 const double *data);

typedef int (*tinyopt_block_sparse_acc_hessian_func_t)(const double *x, int dims, double *cost,
                                                       double *gradient,
                                                       tinyopt_block_sparse_t *hessian,
                                                       void *user_data);

typedef struct tinyopt_block_sparse_problem_t {
  tinyopt_block_sparse_layout_t layout; /* Runtime block layout; dims must match params->dims. */
  tinyopt_block_sparse_acc_hessian_func_t acc_hessian; /* Cost, gradient and Hessian callback. */
  void *user_data;                                     /* User data passed to the callback. */
} tinyopt_block_sparse_problem_t;

/* The set of blocks written by the callback must be constant across evaluations. */
TINYOPT_C_API tinyopt_status_t tinyopt_optimize_block_sparse(
    const tinyopt_params_t *params, const tinyopt_block_sparse_problem_t *problem,
    const tinyopt_options_t *options, tinyopt_summary_t *summary);

#ifdef __cplusplus
}
#endif

#endif /* TINYOPT_C_API_ENABLE_DYNAMIC */

#endif
