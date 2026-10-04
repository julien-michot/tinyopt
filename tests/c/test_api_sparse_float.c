// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stddef.h>

#include <tinyopt/c/c_api_sparse_float.h>

#define N 20

/* f(x) = 0.5 * sum (x_i - t_i)^2 with a diagonal Hessian. */
static int accumulate(const float *x, int dims, float *cost, float *gradient,
                      cholmod_sparse **hessian, cholmod_common *common, void *user_data) {
  const float *target = (const float *)user_data;
  if (dims != N) return 1;
  *cost = 0.0f;
  for (int i = 0; i < N; ++i) {
    const float d = x[i] - target[i];
    *cost += 0.5f * d * d;
    if (gradient != NULL) gradient[i] = d;
  }
  if (hessian == NULL) return 0;

  cholmod_triplet *triplet =
      cholmod_allocate_triplet(N, N, N, 1, CHOLMOD_REAL + CHOLMOD_SINGLE, common);
  if (triplet == NULL) return 1;
  for (int i = 0; i < N; ++i) {
    ((int *)triplet->i)[i] = i;
    ((int *)triplet->j)[i] = i;
    ((float *)triplet->x)[i] = 1.0f;
  }
  triplet->nnz = N;
  *hessian = cholmod_triplet_to_sparse(triplet, N, common);
  cholmod_free_triplet(&triplet, common);
  return *hessian == NULL ? 1 : 0;
}

int main(void) {
  float x[N] = {0};
  float target[N];
  for (int i = 0; i < N; ++i) target[i] = 0.1f * (float)i;
  tinyopt_paramsf_t params = {x, N, NULL};
  tinyopt_sparse_problemf_t problem = {accumulate, target};
  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.log_enabled = 0;
  tinyopt_summary_t summary = {0};

  if (tinyopt_optimize_sparsef(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK)
    return 2;
  for (int i = 0; i < N; ++i)
    if (fabsf(x[i] - target[i]) > 1e-3f) return 3;
  if (summary.final_cost > 1e-5) return 4;
  return 0;
}
