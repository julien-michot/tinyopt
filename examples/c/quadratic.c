// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>

#include <tinyopt/c/c_api.h>

static void plus_eq(double *x, double *dx) { x[0] += dx[0]; }

static int residuals(const double *x, int dims, double *result, int residual_dims,
                     void *user_data) {
  const double target = *(const double *)user_data;
  if (dims != 1 || residual_dims != 1) return 1;
  result[0] = x[0] - target;
  return 0;
}

int main(void) {
  double x[] = {-3.0};
  double target = 0.75;
  tinyopt_params params = {x, 1, plus_eq};
  tinyopt_residuals residual_func = {residuals, 1, &target};
  tinyopt_summary summary;
  const tinyopt_status status = tinyopt_optimize(&params, &residual_func, NULL, &summary);
  if (status != TINYOPT_STATUS_OK) {
    fprintf(stderr, "tinyopt_optimize failed: status=%d\n", (int)status);
    return (int)status;
  }

  printf("x=%.8f cost=%.3e iterations=%d\n", x[0], summary.final_cost, summary.num_iters);
  return 0;
}