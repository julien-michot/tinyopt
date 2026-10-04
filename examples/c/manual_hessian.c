// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stdio.h>

#include <tinyopt/c/c_api.h>

typedef struct Target {
  double values[2];
} Target;

static void plus_eq(double *params, double *delta) {
  for (int i = 0; i < 2; ++i) params[i] += delta[i];
}

static int accumulate_cost_gradient_hessian(const double *params, int dims, double *cost,
                                            double *gradient, double *hessian, void *user_data) {
  const Target *target = (const Target *)user_data;
  if (dims != 2) return 1;
  const double dx = params[0] - target->values[0];
  const double dy = params[1] - target->values[1];
  *cost = 0.5 * (dx * dx + dy * dy);
  if (gradient != NULL) {
    gradient[0] = dx;
    gradient[1] = dy;
  }
  if (hessian != NULL) {
    hessian[0] = 1.0;
    hessian[1] = 0.0;
    hessian[2] = 0.0;
    hessian[3] = 1.0;
  }
  return 0;
}

int main(void) {
  double values[] = {-5.0, 9.0};
  const Target target = {{0.5, -1.5}};
  tinyopt_params_t params = {values, 2, plus_eq};
  tinyopt_problem_t problem = {0};
  problem.type = TINYOPT_EVAL_HESSIAN;
  problem.fn.acc_hessian = accumulate_cost_gradient_hessian;
  problem.user_data = (void *)&target;

  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.hessian_is_full = 1;
  options.log_enabled = 0;
  tinyopt_summary_t summary = {0};
  const tinyopt_status_t status = tinyopt_optimize(&params, &problem, &options, &summary);
  if (status != TINYOPT_STATUS_OK) {
    fprintf(stderr, "Hessian optimization failed: status=%d\n", (int)status);
    return (int)status;
  }
  if (fabs(values[0] - target.values[0]) > 1e-5 || fabs(values[1] - target.values[1]) > 1e-5)
    return 2;

  printf("point=(%.6f, %.6f) cost=%.3g\n", values[0], values[1], summary.final_cost);
  return 0;
}