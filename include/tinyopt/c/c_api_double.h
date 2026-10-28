// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_DOUBLE_H
#define TINYOPT_C_C_API_DOUBLE_H

#include <tinyopt/c/c_api_common.h>
#include <tinyopt/c/c_api_problem_double.h>

#if TINYOPT_C_API_ENABLE_DYNAMIC

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tinyopt_params_t {
  double *x; /* In/out parameter values; updated on success or requested stop. */
  int dims;  /* Number of parameter values; must be positive. */
  void (*plus_eq)(double *x, double *dx); /* Optional update; NULL uses component-wise addition. */
} tinyopt_params_t;

/* Uses default Levenberg-Marquardt options and numerical differentiation. */
TINYOPT_C_API tinyopt_status_t tinyopt_optimize(const tinyopt_params_t *params,
                                                const tinyopt_problem_t *problem,
                                                const tinyopt_options_t *options,
                                                tinyopt_summary_t *summary);

#ifdef __cplusplus
}
#endif

#endif /* TINYOPT_C_API_ENABLE_DYNAMIC */

#include <tinyopt/c/c_api_fixed_double.h>

#endif
