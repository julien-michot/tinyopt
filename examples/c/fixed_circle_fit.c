// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stdio.h>

#include <tinyopt/c/c_api.h>

enum { kPointCount = 8 };

typedef struct CircleData {
  double points[kPointCount][2];
} CircleData;

static int circle_residuals(const double *params, int dims, double *residuals, double **jacobian,
                            int num_residuals, void *user_data) {
  const CircleData *data = (const CircleData *)user_data;
  if (dims != 3 || num_residuals != kPointCount) return 1;

  for (int row = 0; row < kPointCount; ++row) {
    const double dx = params[0] - data->points[row][0];
    const double dy = params[1] - data->points[row][1];
    const double distance = sqrt(dx * dx + dy * dy);
    if (distance == 0.0) return 1;
    residuals[row] = distance - params[2];
    if (*jacobian != NULL) {
      (*jacobian)[row * 3] = dx / distance;
      (*jacobian)[row * 3 + 1] = dy / distance;
      (*jacobian)[row * 3 + 2] = -1.0;
    }
  }
  return 0;
}

int main(void) {
  double values[] = {0.0, 0.0, 2.0};
  const CircleData data = {{{3.98, -0.80},
                            {3.074, 1.074},
                            {1.20, 1.94},
                            {-0.653, 1.053},
                            {-1.53, -0.80},
                            {-0.695, -2.695},
                            {1.20, -3.57},
                            {3.117, -2.717}}};
  tinyopt_params3_t params = {values, NULL};
  tinyopt_problem_t problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = circle_residuals;
  problem.num_residuals = kPointCount;
  problem.user_data = (void *)&data;

  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.log_enabled = 0;
  tinyopt_summary_t summary = {0};
  const tinyopt_status_t status = tinyopt_optimize3(&params, &problem, &options, &summary);
  if (status != TINYOPT_STATUS_OK) {
    fprintf(stderr, "circle fit failed: status=%d\n", (int)status);
    return (int)status;
  }
  if (fabs(values[0] - 1.2) > 0.15 || fabs(values[1] + 0.8) > 0.15 || fabs(values[2] - 2.7) > 0.15)
    return 1;

  printf("center=(%.5f, %.5f) radius=%.5f cost=%.3g\n", values[0], values[1], values[2],
         summary.final_cost);
  return 0;
}