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

/* Return 0 after filling all residual_dims entries, or nonzero on callback failure. */
typedef int (*tinyopt_residual_func)(const double *x, int dims, double *residuals,
                                     int residual_dims, void *user_data);

typedef struct tinyopt_residuals {
  tinyopt_residual_func evaluate;
  int dims;
  void *user_data;
} tinyopt_residuals;

/* Uses default Levenberg-Marquardt options and numerical differentiation. */
TINYOPT_C_API tinyopt_status tinyopt_optimize(const tinyopt_params *params,
                                              const tinyopt_residuals *residuals,
                                              const tinyopt_options *options,
                                              tinyopt_summary *summary);

#ifdef __cplusplus
}
#endif

#include <tinyopt/c/c_api_fixed_double.h>

#endif