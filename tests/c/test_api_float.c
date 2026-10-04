// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stddef.h>

#include <tinyopt/c/c_api_float.h>

static void plus_eq(float *x, float *dx) { x[0] += dx[0]; }

static int residuals(const float *x, int dims, float *result, float **jacobian, int residual_dims,
                     void *user_data) {
  const float target = *(const float *)user_data;
  if (dims != 1 || residual_dims != 1) return 1;
  result[0] = x[0] - target;
  if (*jacobian != NULL) (*jacobian)[0] = 1.0f;
  return 0;
}

static int stop_callback2(float error, const float *x, const float *dx, int dims, void *user_data) {
  (void)error;
  (void)dx;
  if (dims != 1 || x == NULL) return 0;
  ++*(int *)user_data;
  return 1;
}

int main(void) {
  float x[] = {-2.0f};
  float target = 0.5f;
  tinyopt_paramsf_t params = {x, 1, plus_eq};
  tinyopt_problemf_t problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = residuals;
  problem.num_residuals = 1;
  problem.user_data = &target;
  tinyopt_summary_t summary = {0};

  if (tinyopt_optimizef(&params, &problem, NULL, &summary) != TINYOPT_STATUS_OK) return 1;
  if (fabsf(x[0] - target) > 1e-4f) return 2;
  if (summary.num_residuals != 1 || summary.final_cost > 1e-5) return 3;

  x[0] = -2.0f;
  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 4;
  int callback_count = 0;
  options.stop_callback2 = stop_callback2;
  options.stop_callback2_user_data = &callback_count;
  if (tinyopt_optimizef(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 5;
  if (callback_count != 1) return 6;
  return 0;
}