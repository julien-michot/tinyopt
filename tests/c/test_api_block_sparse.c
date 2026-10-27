// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stddef.h>

#include <tinyopt/c/c_api.h>

#define DIMS 5
#define FIRST_BLOCK 2
#define SECOND_BLOCK 3

static const double kOffDiagonal[FIRST_BLOCK * SECOND_BLOCK] = {0.3, 0.2, 0.1, 0.0, 0.0, 0.4};

static void gradient(const double *x, const double *target, double *out) {
  const double d0[FIRST_BLOCK] = {x[0] - target[0], x[1] - target[1]};
  const double d1[SECOND_BLOCK] = {x[2] - target[2], x[3] - target[3], x[4] - target[4]};
  for (int i = 0; i < FIRST_BLOCK; ++i) {
    out[i] = 4.0 * d0[i];
    for (int j = 0; j < SECOND_BLOCK; ++j) out[i] += kOffDiagonal[i + j * FIRST_BLOCK] * d1[j];
  }
  for (int j = 0; j < SECOND_BLOCK; ++j) {
    out[2 + j] = 5.0 * d1[j];
    for (int i = 0; i < FIRST_BLOCK; ++i) out[2 + j] += kOffDiagonal[i + j * FIRST_BLOCK] * d0[i];
  }
}

static int accumulate(const double *x, int dims, double *cost, double *grad,
                      tinyopt_block_sparse_t *hessian, void *user_data) {
  const double *target = (const double *)user_data;
  if (dims != DIMS) return 1;
  double g[DIMS] = {0};
  gradient(x, target, g);
  if (grad != NULL) {
    for (int i = 0; i < DIMS; ++i) grad[i] = g[i];
  }
  double value = 0.0;
  for (int i = 0; i < DIMS; ++i) value += (x[i] - target[i]) * g[i];
  *cost = 0.5 * value;

  if (hessian != NULL) {
    int rows = 0;
    int cols = 0;
    if (tinyopt_block_sparse_dims(hessian) != DIMS) return 1;
    if (tinyopt_block_sparse_num_block_types(hessian) != 2) return 1;
    if (tinyopt_block_sparse_block_size(hessian, 0) != FIRST_BLOCK) return 1;
    if (tinyopt_block_sparse_num_blocks(hessian, 1) != 1) return 1;
    if (tinyopt_block_sparse_block_offset(hessian, 1) != FIRST_BLOCK) return 1;
    if (tinyopt_block_sparse_block_dims(hessian, 0, 1, &rows, &cols) != 0) return 1;
    if (rows != FIRST_BLOCK || cols != SECOND_BLOCK) return 1;
    double first[FIRST_BLOCK * FIRST_BLOCK] = {0};
    double second[SECOND_BLOCK * SECOND_BLOCK] = {0};
    for (int i = 0; i < FIRST_BLOCK; ++i) first[i + i * FIRST_BLOCK] = 4.0;
    for (int i = 0; i < SECOND_BLOCK; ++i) second[i + i * SECOND_BLOCK] = 5.0;
    if (tinyopt_block_sparse_set_block(hessian, 0, 0, first) != 0) return 1;
    if (tinyopt_block_sparse_set_block(hessian, 1, 1, second) != 0) return 1;
    if (tinyopt_block_sparse_set_block(hessian, 0, 1, kOffDiagonal) != 0) return 1;
  }
  return 0;
}

int main(void) {
  const int sizes[2] = {FIRST_BLOCK, SECOND_BLOCK};
  const int counts[2] = {1, 1};
  double x[DIMS] = {0};
  double target[DIMS] = {1.0, -1.0, 0.5, 2.0, -0.5};

  tinyopt_params_t params = {x, DIMS, NULL};
  tinyopt_block_sparse_layout_t layout = {sizes, counts, 2};
  tinyopt_block_sparse_problem_t problem = {layout, accumulate, target};
  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.max_iters = 50;
  options.exp_final_cov = 1;
  double covariance[DIMS * DIMS];
  options.final_covariance_buffer = covariance;
  options.final_covariance_capacity = DIMS * DIMS;

  tinyopt_summary_t summary = {0};
  if (tinyopt_optimize_block_sparse(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK)
    return 1;
  if (summary.final_covariance_dims != DIMS) return 1;

  double g[DIMS] = {0};
  gradient(x, target, g);
  double squared = 0.0;
  for (int i = 0; i < DIMS; ++i) squared += g[i] * g[i];
  if (sqrt(squared) > 1e-6) return 1;
  for (int i = 0; i < DIMS; ++i) {
    if (fabs(x[i] - target[i]) > 1e-6) return 1;
  }
  return 0;
}
