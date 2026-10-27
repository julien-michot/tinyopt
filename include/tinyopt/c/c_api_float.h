// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_FLOAT_H
#define TINYOPT_C_C_API_FLOAT_H

#include <tinyopt/c/c_api_common.h>
#include <tinyopt/c/c_api_problem_float.h>

#if TINYOPT_C_API_ENABLE_DYNAMIC && TINYOPT_C_API_ENABLE_FLOAT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tinyopt_paramsf_t {
  float *x; /* In/out parameter values; updated on success or requested stop. */
  int dims; /* Number of parameter values; must be positive. */
  void (*plus_eq)(float *x, float *dx); /* Optional update; NULL uses component-wise addition. */
} tinyopt_paramsf_t;

/* Uses default Levenberg-Marquardt options and numerical differentiation. */
TINYOPT_C_API tinyopt_status_t tinyopt_optimizef(const tinyopt_paramsf_t *params,
                                                 const tinyopt_problemf_t *problem,
                                                 const tinyopt_options_t *options,
                                                 tinyopt_summary_t *summary);

#ifdef __cplusplus
}
#endif

#endif /* TINYOPT_C_API_ENABLE_DYNAMIC && TINYOPT_C_API_ENABLE_FLOAT */

#include <tinyopt/c/c_api_fixed_float.h>

#endif
