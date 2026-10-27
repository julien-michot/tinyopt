// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stddef.h>

#include <tinyopt/c/c_api.h>
#include <tinyopt/c/c_api_sparse.h>

#if TINYOPT_C_API_ENABLE_DYNAMIC
#error "Dynamic C APIs must be disabled in this configuration"
#endif

static int residuals(const double *x, int dims, double *out, double **jacobian, int residual_dims,
                     void *user_data) {
  (void)user_data;
  if (dims != 2 || residual_dims != 2) return 1;
  out[0] = x[0] - 1.0;
  out[1] = x[1] + 2.0;
  if (*jacobian != NULL) {
    (*jacobian)[0] = 1.0;
    (*jacobian)[1] = 0.0;
    (*jacobian)[2] = 0.0;
    (*jacobian)[3] = 1.0;
  }
  return 0;
}

int main(void) {
  double x[2] = {0.0, 0.0};
  tinyopt_params2_t params = {x, NULL};
  tinyopt_problem_t problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = residuals;
  problem.num_residuals = 2;

  const tinyopt_status_t status = tinyopt_optimize2(&params, &problem, NULL, NULL);
  if (status != TINYOPT_STATUS_OK) return 1;
  return (fabs(x[0] - 1.0) < 1e-6 && fabs(x[1] + 2.0) < 1e-6) ? 0 : 1;
}
