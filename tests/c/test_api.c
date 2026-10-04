// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <tinyopt/c/c_api.h>

typedef struct TestData {
  double target[2];
  int provide_jacobian;
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

static int residuals(const double *x, int dims, double *result, double **jacobian,
                     int residual_dims, void *user_data) {
  const TestData *data = (const TestData *)user_data;
  if (dims != 2 || residual_dims != 2) return 1;
  result[0] = x[0] - data->target[0];
  result[1] = x[1] - data->target[1];
  if (!data->provide_jacobian) {
    *jacobian = NULL;
  } else if (*jacobian != NULL) {
    (*jacobian)[0] = 1.0;
    (*jacobian)[1] = 0.0;
    (*jacobian)[2] = 0.0;
    (*jacobian)[3] = 1.0;
  }
  return 0;
}

static int rosenbrock(const double *x, int dims, double *result, double **jacobian,
                      int residual_dims, void *user_data) {
  (void)user_data;
  if (dims != 2 || residual_dims != 2) return 1;
  result[0] = 10.0 * (x[1] - x[0] * x[0]);
  result[1] = 1.0 - x[0];
  *jacobian = NULL;
  return 0;
}

static int failing_residuals(const double *x, int dims, double *result, double **jacobian,
                             int residual_dims, void *user_data) {
  (void)x;
  (void)dims;
  (void)result;
  (void)jacobian;
  (void)residual_dims;
  (void)user_data;
  return 1;
}

static int cost_only(const double *x, int dims, double *cost, void *user_data) {
  const TestData *data = (const TestData *)user_data;
  if (dims != 2) return 1;
  const double dx = x[0] - data->target[0];
  const double dy = x[1] - data->target[1];
  *cost = 0.5 * (dx * dx + dy * dy);
  return 0;
}

static int accumulate_hessian(const double *x, int dims, double *cost, double *gradient,
                              double *hessian, void *user_data) {
  const TestData *data = (const TestData *)user_data;
  if (dims != 2) return 1;
  const double dx = x[0] - data->target[0];
  const double dy = x[1] - data->target[1];
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

#if defined(TINYOPT_TEST_ENABLE_GRADIENT_DESCENT)
static int accumulate_gradient(const double *x, int dims, double *cost, double *gradient,
                               void *user_data) {
  const TestData *data = (const TestData *)user_data;
  if (dims != 2) return 1;
  const double dx = x[0] - data->target[0];
  const double dy = x[1] - data->target[1];
  *cost = 0.5 * (dx * dx + dy * dy);
  if (gradient != NULL) {
    gradient[0] = dx;
    gradient[1] = dy;
  }
  return 0;
}
#endif

int main(void) {
  double x[] = {10.0, -5.0};
  TestData data = {{2.0, 4.0}, 0};
  tinyopt_params_t params = {x, 2, plus_eq};
  tinyopt_problem_t problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = residuals;
  problem.num_residuals = 2;
  problem.user_data = &data;
  tinyopt_summary_t summary = {0};

  tinyopt_options_t options;
  if (tinyopt_options_default(NULL) != TINYOPT_STATUS_INVALID_ARGUMENT) return 1;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.lm_damping_init = 1e-4f;
  options.log_enabled = 0;
  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 1;
  if (fabs(x[0] - data.target[0]) > 1e-5 || fabs(x[1] - data.target[1]) > 1e-5) return 2;
  if (summary.final_cost > 1e-10 || summary.num_residuals != 2 ||
      summary.used_numerical_differentiation != 1) {
    fprintf(stderr, "cost=%g residuals=%d num_diff=%d\n", summary.final_cost, summary.num_residuals,
            summary.used_numerical_differentiation);
    return 3;
  }

  x[0] = 10.0;
  x[1] = -5.0;
  data.provide_jacobian = 1;
  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 23;
  if (summary.used_numerical_differentiation != 0) return 24;
  data.provide_jacobian = 0;

  if (options.max_iters != 50 || options.solver_type != TINYOPT_SOLVER_LEVENBERG_MARQUARDT ||
      options.linear_solver != TINYOPT_LINEAR_SOLVER_LDLT || options.log_error_symbol == NULL)
    return 4;

  x[0] = 10.0;
  x[1] = -5.0;
  int callback_count = 0;
  options.stop_callback = stop_after_first_iteration;
  options.stop_callback_user_data = &callback_count;
  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 5;
  if (callback_count != 1) return 6;

  x[0] = 10.0;
  x[1] = -5.0;
  options.solver_type = TINYOPT_SOLVER_GRADIENT_DESCENT;
  options.gd_learning_rate = 0.5f;
  options.stop_callback = NULL;
  problem.type = TINYOPT_EVAL_COST_ONLY;
  problem.fn.cost = cost_only;
  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 7;
  if (fabs(x[0] - data.target[0]) > 1e-5 || fabs(x[1] - data.target[1]) > 1e-5 ||
      summary.used_numerical_differentiation == 0) {
    fprintf(stderr, "cost-only x=%g,%g target=%g,%g diff=%d stop=%d cost=%g\n", x[0], x[1],
            data.target[0], data.target[1], summary.used_numerical_differentiation,
            summary.stop_reason, summary.final_cost);
    return 8;
  }

  x[0] = 10.0;
  x[1] = -5.0;
  options.solver_type = TINYOPT_SOLVER_LEVENBERG_MARQUARDT;
  problem.type = TINYOPT_EVAL_HESSIAN;
  problem.fn.acc_hessian = accumulate_hessian;
  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 9;
  if (fabs(x[0] - data.target[0]) > 1e-5 || fabs(x[1] - data.target[1]) > 1e-5 ||
      summary.used_numerical_differentiation != 0)
    return 10;

#if defined(TINYOPT_TEST_ENABLE_GRADIENT_DESCENT)
  x[0] = 10.0;
  x[1] = -5.0;
  problem.type = TINYOPT_EVAL_GRADIENT;
  problem.fn.acc_grad = accumulate_gradient;
  options.solver_type = TINYOPT_SOLVER_GRADIENT_DESCENT;
  options.gd_learning_rate = 0.5f;
  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 11;
  if (fabs(x[0] - data.target[0]) > 1e-5 || fabs(x[1] - data.target[1]) > 1e-5) return 12;
  options.solver_type = TINYOPT_SOLVER_LEVENBERG_MARQUARDT;
#endif

  x[0] = 7.0;
  x[1] = 8.0;
  tinyopt_problem_t failing_problem = {0};
  failing_problem.type = TINYOPT_EVAL_RESIDUALS;
  failing_problem.fn.residuals = failing_residuals;
  failing_problem.num_residuals = 2;
  problem = failing_problem;
  if (tinyopt_optimize(&params, &failing_problem, NULL, &summary) != TINYOPT_STATUS_USER_STOPPED)
    return 13;
  if (x[0] != 7.0 || x[1] != 8.0) return 14;

  params.plus_eq = NULL;
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = residuals;
  problem.num_residuals = 2;
  problem.user_data = &data;
  data.provide_jacobian = 0;
  x[0] = 10.0;
  x[1] = -5.0;
  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 15;
  if (fabs(x[0] - data.target[0]) > 1e-5 || fabs(x[1] - data.target[1]) > 1e-5) return 25;

  params.plus_eq = plus_eq;
  params.dims = 0;
  if (tinyopt_optimize(&params, &problem, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT)
    return 16;
  params.dims = -1;
  if (tinyopt_optimize(&params, &problem, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT)
    return 17;
  params.dims = 2;

  tinyopt_problem_t invalid_problem = failing_problem;
  invalid_problem.num_residuals = 0;
  if (tinyopt_optimize(&params, &invalid_problem, NULL, &summary) !=
      TINYOPT_STATUS_INVALID_ARGUMENT)
    return 18;
  invalid_problem.num_residuals = -1;
  if (tinyopt_optimize(&params, &invalid_problem, NULL, &summary) !=
      TINYOPT_STATUS_INVALID_ARGUMENT)
    return 19;
  invalid_problem = failing_problem;
  invalid_problem.fn.residuals = NULL;
  if (tinyopt_optimize(&params, &invalid_problem, NULL, &summary) !=
      TINYOPT_STATUS_INVALID_ARGUMENT)
    return 20;
  if (tinyopt_optimize(NULL, &problem, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT)
    return 21;
  if (tinyopt_optimize(&params, NULL, NULL, &summary) != TINYOPT_STATUS_INVALID_ARGUMENT) return 16;

  params.dims = 2;
  params.plus_eq = plus_eq;
  x[0] = data.target[0];
  x[1] = data.target[1];
  problem = failing_problem;
  problem.fn.residuals = residuals;
  problem.user_data = &data;
  problem.num_residuals = 2;
  if (tinyopt_optimize(&params, &problem, NULL, NULL) != TINYOPT_STATUS_OK) return 22;
  data.provide_jacobian = 1;
  if (tinyopt_optimize(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK) return 23;
  if (summary.used_numerical_differentiation != 0) return 24;

  /* Regression: rejected LM steps re-evaluate residuals without a Hessian (dynamic size). */
  double rosen[] = {-1.2, 1.0};
  tinyopt_params_t rosen_params = {rosen, 2, NULL};
  tinyopt_problem_t rosen_problem = {0};
  rosen_problem.type = TINYOPT_EVAL_RESIDUALS;
  rosen_problem.fn.residuals = rosenbrock;
  rosen_problem.num_residuals = 2;
  options.max_iters = 100;
  if (tinyopt_optimize(&rosen_params, &rosen_problem, &options, &summary) != TINYOPT_STATUS_OK)
    return 26;
  if (summary.num_failures == 0 || fabs(rosen[0] - 1.0) > 1e-3 || fabs(rosen[1] - 1.0) > 1e-3)
    return 27;

  return 0;
}