// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stddef.h>

#include <tinyopt/c/c_api_sparse.h>

#define N 50

typedef struct Chain {
  double target[N];
  int lower;      /* Fill the lower triangle (stype = -1) instead of the upper one. */
  int stop_after; /* Return nonzero after this many evaluations when positive. */
  int evaluations;
} Chain;

/* f(x) = 0.5 * sum (x_i - t_i)^2 + 0.5 * sum (x_i - x_{i+1})^2, with a tridiagonal Hessian. */
static double chain_gradient(const double *x, const double *target, double *gradient) {
  double cost = 0.0;
  for (int i = 0; i < N; ++i) {
    const double d = x[i] - target[i];
    cost += 0.5 * d * d;
    if (gradient != NULL) gradient[i] += d;
    if (i + 1 < N) {
      const double e = x[i] - x[i + 1];
      cost += 0.5 * e * e;
      if (gradient != NULL) {
        gradient[i] += e;
        gradient[i + 1] -= e;
      }
    }
  }
  return cost;
}

static int accumulate(const double *x, int dims, double *cost, double *gradient,
                      cholmod_sparse **hessian, cholmod_common *common, void *user_data) {
  Chain *chain = (Chain *)user_data;
  if (dims != N) return 1;
  if (chain->stop_after > 0 && ++chain->evaluations > chain->stop_after) return 1;
  *cost = chain_gradient(x, chain->target, gradient);
  if (hessian == NULL) return 0;

  const int stype = chain->lower ? -1 : 1;
  cholmod_triplet *triplet =
      cholmod_allocate_triplet(N, N, 2 * N, stype, CHOLMOD_REAL + CHOLMOD_DOUBLE, common);
  if (triplet == NULL) return 1;
  int *rows = (int *)triplet->i;
  int *cols = (int *)triplet->j;
  double *values = (double *)triplet->x;
  size_t count = 0;
  for (int i = 0; i < N; ++i) {
    rows[count] = cols[count] = i;
    values[count++] = (i == 0 || i == N - 1) ? 2.0 : 3.0;
    if (i + 1 < N) {
      rows[count] = chain->lower ? i + 1 : i;
      cols[count] = chain->lower ? i : i + 1;
      values[count++] = -1.0;
    }
  }
  triplet->nnz = count;
  *hessian = cholmod_triplet_to_sparse(triplet, count, common);
  cholmod_free_triplet(&triplet, common);
  return *hessian == NULL ? 1 : 0;
}

static int check_solution(const double *x, const Chain *chain) {
  double gradient[N] = {0};
  chain_gradient(x, chain->target, gradient);
  double norm = 0.0;
  for (int i = 0; i < N; ++i) norm += gradient[i] * gradient[i];
  return sqrt(norm) < 1e-5 ? 0 : 1;
}

int main(void) {
  Chain chain = {{0}, 0, 0, 0};
  for (int i = 0; i < N; ++i) chain.target[i] = sin(0.3 * i);

  tinyopt_sparse_problem_t problem = {accumulate, &chain};
  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.log_enabled = 0;

  for (int lower = 0; lower < 2; ++lower) {
    double x[N] = {0};
    chain.lower = lower;
    tinyopt_params_t params = {x, N, NULL};
    tinyopt_summary_t summary = {0};
    if (tinyopt_optimize_sparse(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK)
      return 2;
    if (check_solution(x, &chain) != 0) return 3;
    if (summary.num_iters <= 0) return 4;
  }

  /* Gauss-Newton also works. */
  {
    double x[N] = {0};
    chain.lower = 0;
    options.solver_type = TINYOPT_SOLVER_GAUSS_NEWTON;
    tinyopt_params_t params = {x, N, NULL};
    if (tinyopt_optimize_sparse(&params, &problem, &options, NULL) != TINYOPT_STATUS_OK) return 5;
    if (check_solution(x, &chain) != 0) return 6;
    options.solver_type = TINYOPT_SOLVER_LEVENBERG_MARQUARDT;
  }

  /* A callback can stop the optimization. */
  {
    double x[N] = {0};
    chain.stop_after = 1;
    tinyopt_params_t params = {x, N, NULL};
    if (tinyopt_optimize_sparse(&params, &problem, &options, NULL) != TINYOPT_STATUS_USER_STOPPED)
      return 7;
    chain.stop_after = 0;
  }

  /* Invalid arguments. */
  {
    double x[N] = {0};
    tinyopt_params_t params = {x, N, NULL};
    tinyopt_sparse_problem_t empty = {NULL, NULL};
    if (tinyopt_optimize_sparse(&params, NULL, &options, NULL) != TINYOPT_STATUS_INVALID_ARGUMENT)
      return 8;
    if (tinyopt_optimize_sparse(&params, &empty, &options, NULL) != TINYOPT_STATUS_INVALID_ARGUMENT)
      return 9;
    options.solver_type = TINYOPT_SOLVER_BFGS;
    if (tinyopt_optimize_sparse(&params, &problem, &options, NULL) !=
        TINYOPT_STATUS_INVALID_ARGUMENT)
      return 10;
  }
  return 0;
}
