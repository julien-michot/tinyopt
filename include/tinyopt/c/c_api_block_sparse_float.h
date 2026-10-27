// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_BLOCK_SPARSE_FLOAT_H
#define TINYOPT_C_C_API_BLOCK_SPARSE_FLOAT_H

#include <tinyopt/c/c_api_common.h>

#if TINYOPT_C_API_ENABLE_DYNAMIC && TINYOPT_C_API_ENABLE_FLOAT

#include <tinyopt/c/c_api_float.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed handle, valid only while an accumulation callback runs. */
typedef struct tinyopt_block_sparsef_t tinyopt_block_sparsef_t;

typedef struct tinyopt_block_sparse_layoutf_t {
  const int *block_sizes;
  const int *block_counts;
  int num_block_types;
} tinyopt_block_sparse_layoutf_t;

TINYOPT_C_API int tinyopt_block_sparsef_dims(const tinyopt_block_sparsef_t *hessian);
TINYOPT_C_API int tinyopt_block_sparsef_num_block_types(const tinyopt_block_sparsef_t *hessian);
TINYOPT_C_API int tinyopt_block_sparsef_block_size(const tinyopt_block_sparsef_t *hessian,
                                                   int type);
TINYOPT_C_API int tinyopt_block_sparsef_num_blocks(const tinyopt_block_sparsef_t *hessian,
                                                   int type);
TINYOPT_C_API int tinyopt_block_sparsef_block_offset(const tinyopt_block_sparsef_t *hessian,
                                                     int type);
TINYOPT_C_API int tinyopt_block_sparsef_block_dims(const tinyopt_block_sparsef_t *hessian, int row,
                                                   int col, int *rows, int *cols);
TINYOPT_C_API int tinyopt_block_sparsef_set_block(tinyopt_block_sparsef_t *hessian, int row,
                                                  int col, const float *data);

typedef int (*tinyopt_block_sparse_acc_hessian_funcf_t)(const float *x, int dims, float *cost,
                                                        float *gradient,
                                                        tinyopt_block_sparsef_t *hessian,
                                                        void *user_data);

typedef struct tinyopt_block_sparse_problemf_t {
  tinyopt_block_sparse_layoutf_t layout;
  tinyopt_block_sparse_acc_hessian_funcf_t acc_hessian;
  void *user_data;
} tinyopt_block_sparse_problemf_t;

TINYOPT_C_API tinyopt_status_t tinyopt_optimize_block_sparsef(
    const tinyopt_paramsf_t *params, const tinyopt_block_sparse_problemf_t *problem,
    const tinyopt_options_t *options, tinyopt_summary_t *summary);

#ifdef __cplusplus
}
#endif

#endif /* TINYOPT_C_API_ENABLE_DYNAMIC && TINYOPT_C_API_ENABLE_FLOAT */

#endif
