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

/* Return 0 after filling all residual_dims entries, or nonzero on callback failure. */
typedef int (*tinyopt_residual_funcf)(const float *x, int dims, float *residuals, int residual_dims,
                                      void *user_data);

typedef struct tinyopt_residualsf {
  tinyopt_residual_funcf evaluate;
  int dims;
  void *user_data;
} tinyopt_residualsf;

/* Uses default Levenberg-Marquardt options and numerical differentiation. */
TINYOPT_C_API tinyopt_status tinyopt_optimizef(const tinyopt_paramsf *params,
                                               const tinyopt_residualsf *residuals,
                                               const tinyopt_options *options,
                                               tinyopt_summary *summary);

#ifdef __cplusplus
}
#endif

#include <tinyopt/c/c_api_fixed_float.h>

#endif

#endif