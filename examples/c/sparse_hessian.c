// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

// Fits a smooth 1D signal to noisy samples: minimize 0.5 * sum (x_i - y_i)^2 + 0.5 * w * sum
// (x_i - x_{i+1})^2. The Hessian is tridiagonal, provided as a SuiteSparse cholmod_sparse matrix.

#include <math.h>
#include <stdio.h>

#include <tinyopt/c/c_api_sparse.h>

#define N 200

typedef struct Signal {
  double samples[N];
  double smoothness;
} Signal;

static int accumulate(const double *x, int dims, double *cost, double *gradient,
                      cholmod_sparse **hessian, cholmod_common *common, void *user_data) {
  const Signal *signal = (const Signal *)user_data;
  if (dims != N) return 1;

  const double w = signal->smoothness;
  *cost = 0.0;
  for (int i = 0; i < N; ++i) {
    const double d = x[i] - signal->samples[i];
    *cost += 0.5 * d * d;
    if (gradient != NULL) gradient[i] += d;
    if (i + 1 < N) {
      const double e = x[i] - x[i + 1];
      *cost += 0.5 * w * e * e;
      if (gradient != NULL) {
        gradient[i] += w * e;
        gradient[i + 1] -= w * e;
      }
    }
  }
  if (hessian == NULL) return 0;

  // Upper triangle (stype = 1) of the tridiagonal Hessian
  cholmod_triplet *triplet =
      cholmod_allocate_triplet(N, N, 2 * N, 1, CHOLMOD_REAL + CHOLMOD_DOUBLE, common);
  if (triplet == NULL) return 1;
  int *rows = (int *)triplet->i;
  int *cols = (int *)triplet->j;
  double *values = (double *)triplet->x;
  size_t count = 0;
  for (int i = 0; i < N; ++i) {
    rows[count] = cols[count] = i;
    values[count++] = 1.0 + w * ((i == 0 || i == N - 1) ? 1.0 : 2.0);
    if (i + 1 < N) {
      rows[count] = i;
      cols[count] = i + 1;
      values[count++] = -w;
    }
  }
  triplet->nnz = count;
  *hessian = cholmod_triplet_to_sparse(triplet, count, common);
  cholmod_free_triplet(&triplet, common);
  return *hessian == NULL ? 1 : 0;
}

int main(void) {
  Signal signal = {{0}, 5.0};
  double x[N] = {0};
  for (int i = 0; i < N; ++i) signal.samples[i] = sin(0.05 * i) + 0.2 * sin(7.0 * i);

  tinyopt_params_t params = {x, N, NULL};
  tinyopt_sparse_problem_t problem = {accumulate, &signal};
  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.log_enabled = 0;
  tinyopt_summary_t summary = {0};

  const tinyopt_status_t status = tinyopt_optimize_sparse(&params, &problem, &options, &summary);
  if (status != TINYOPT_STATUS_OK) {
    fprintf(stderr, "Sparse optimization failed: status=%d\n", (int)status);
    return (int)status;
  }
  printf("x[0]=%.4f x[%d]=%.4f cost=%.4g iterations=%d\n", x[0], N - 1, x[N - 1],
         summary.final_cost, summary.num_iters);
  return 0;
}
