// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <tinyopt/c/c_api.h>

typedef struct TestData {
  double target[2];
} TestData;

static int stop_after_first_iteration(double error, double step_norm_squared,
                                      double gradient_norm_squared, void *user_data) {
  (void)error;
  (void)step_norm_squared;
  (void)gradient_norm_squared;
  ++*(int *)user_data;
  return 1;
}

static void plus_eq(double *x, double *dx) {
  x[0] += dx[0];
  x[1] += dx[1];
}

static int residuals(const double *x, int dims, double *result, int residual_dims,
                     void *user_data) {
  const TestData *data = (const TestData *)user_data;
  if (dims != 2 || residual_dims != 2) return 1;
  result[0] = x[0] - data->target[0];
  result[1] = x[1] - data->target[1];
  return 0;
}

static int failing_residuals(const double *x, int dims, double *result, int residual_dims,
                             void *user_data) {
  (void)x;
  (void)dims;
  (void)result;
  (void)residual_dims;
  (void)user_data;
  return 1;
}

int main(void) {
  double x[] = {10.0, -5.0};
  TestData data = {{2.0, 4.0}};
  tinyopt_params params = {x, 2, plus_eq};
  tinyopt_residuals residual_func = {residuals, 2, &data};
  tinyopt_summary summary = {0};

  tinyopt_options options;
  if (tinyopt_options_default(NULL) != TINYOPT_STATUS_INVALID_ARGUMENT) return 1;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.lm_damping_init = 1e-4f;
  options.log_enabled = 0;
  if (tinyopt_optimize(&params, &residual_func, &options, &summary) != TINYOPT_STATUS_OK) return 1;
  if (fabs(x[0] - data.target[0]) > 1e-5 || fabs(x[1] - data.target[1]) > 1e-5) return 2;
  if (summary.final_cost > 1e-10 || summary.num_residuals != 2 ||
      summary.used_numerical_differentiation != 1) {
    fprintf(stderr, "cost=%g residuals=%d num_diff=%d\n", summary.final_cost, summary.num_residuals,
            summary.used_numerical_differentiation);
    return 3;
  }

  if (options.max_iters != 50 || options.solver_type != TINYOPT_SOLVER_LEVENBERG_MARQUARDT ||
      options.linear_solver != TINYOPT_LINEAR_SOLVER_LDLT || options.log_error_symbol == NULL)
    return 4;

  x[0] = 10.0;
  x[1] = -5.0;
  int callback_count = 0;
  options.stop_callback = stop_after_first_iteration;
  options.stop_callback_user_data = &callback_count;
  if (tinyopt_optimize(&params, &residual_func, &options, &summary) != TINYOPT_STATUS_OK) return 5;
  if (callback_count != 1) return 6;

  x[0] = 7.0;
  x[1] = 8.0;
  tinyopt_residuals failing_function = {failing_residuals, 2, NULL};
  if (tinyopt_optimize(&params, &failing_function, NULL, &summary) !=
      TINYOPT_STATUS_RESIDUAL_CALLBACK_FAILED)
    return 7;
  if (x[0] != 7.0 || x[1] != 8.0) return 8;

  params.plus_eq = NULL;
  if (tinyopt_optimize(&params, &residual_func, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT)
    return 9;

  params.plus_eq = plus_eq;
  params.dims = 0;
  if (tinyopt_optimize(&params, &residual_func, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT)
    return 10;
  params.dims = -1;
  if (tinyopt_optimize(&params, &residual_func, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT)
    return 11;
  params.dims = 2;

  tinyopt_residuals invalid_residuals = residual_func;
  invalid_residuals.dims = 0;
  if (tinyopt_optimize(&params, &invalid_residuals, NULL, &summary) !=
      TINYOPT_STATUS_INVALID_ARGUMENT)
    return 12;
  invalid_residuals.dims = -1;
  if (tinyopt_optimize(&params, &invalid_residuals, NULL, &summary) !=
      TINYOPT_STATUS_INVALID_ARGUMENT)
    return 13;
  invalid_residuals = residual_func;
  invalid_residuals.evaluate = NULL;
  if (tinyopt_optimize(&params, &invalid_residuals, NULL, &summary) !=
      TINYOPT_STATUS_INVALID_ARGUMENT)
    return 14;
  if (tinyopt_optimize(NULL, &residual_func, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT)
    return 15;
  if (tinyopt_optimize(&params, NULL, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT) return 16;

  params.dims = 2;
  params.plus_eq = plus_eq;
  x[0] = data.target[0];
  x[1] = data.target[1];
  if (tinyopt_optimize(&params, &residual_func, NULL, NULL) != TINYOPT_STATUS_OK) return 17;

  return 0;
}