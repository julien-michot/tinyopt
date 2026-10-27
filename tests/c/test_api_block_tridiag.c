// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stddef.h>

#include <tinyopt/c/c_api.h>

#define NUM_BLOCKS 4
#define BLOCK_SIZE 2
#define DIMS (NUM_BLOCKS * BLOCK_SIZE)

static double gradient_norm(const double *x, const double *target) {
  double gradient[DIMS] = {0};
  for (int i = 0; i < DIMS; ++i) gradient[i] = x[i] - target[i];
  for (int block = 0; block + 1 < NUM_BLOCKS; ++block) {
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      const double difference = x[block * BLOCK_SIZE + j] - x[(block + 1) * BLOCK_SIZE + j];
      gradient[block * BLOCK_SIZE + j] += difference;
      gradient[(block + 1) * BLOCK_SIZE + j] -= difference;
    }
  }
  double squared = 0.0;
  for (int i = 0; i < DIMS; ++i) squared += gradient[i] * gradient[i];
  return sqrt(squared);
}

static int accumulate(const double *x, int dims, double *cost, double *gradient,
                      tinyopt_block_tridiag_t *hessian, void *user_data) {
  const double *target = (const double *)user_data;
  if (dims != DIMS) return 1;
  double value = 0.0;
  if (gradient != NULL) {
    for (int i = 0; i < DIMS; ++i) gradient[i] = x[i] - target[i];
  }
  for (int i = 0; i < DIMS; ++i) {
    const double difference = x[i] - target[i];
    value += 0.5 * difference * difference;
  }
  for (int block = 0; block + 1 < NUM_BLOCKS; ++block) {
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      const double difference = x[block * BLOCK_SIZE + j] - x[(block + 1) * BLOCK_SIZE + j];
      value += 0.5 * difference * difference;
      if (gradient != NULL) {
        gradient[block * BLOCK_SIZE + j] += difference;
        gradient[(block + 1) * BLOCK_SIZE + j] -= difference;
      }
    }
  }
  *cost = value;

  if (hessian != NULL) {
    if (tinyopt_block_tridiag_dims(hessian) != DIMS) return 1;
    if (tinyopt_block_tridiag_block_size(hessian) != BLOCK_SIZE) return 1;
    if (tinyopt_block_tridiag_num_blocks(hessian) != NUM_BLOCKS) return 1;
    double diagonal[BLOCK_SIZE * BLOCK_SIZE] = {0};
    double off_diagonal[BLOCK_SIZE * BLOCK_SIZE] = {0};
    for (int i = 0; i < NUM_BLOCKS; ++i) {
      const double weight = 1.0 + (i > 0 ? 1.0 : 0.0) + (i + 1 < NUM_BLOCKS ? 1.0 : 0.0);
      for (int j = 0; j < BLOCK_SIZE; ++j) diagonal[j + j * BLOCK_SIZE] = weight;
      if (tinyopt_block_tridiag_set_block(hessian, i, i, diagonal) != 0) return 1;
      if (i + 1 < NUM_BLOCKS) {
        for (int j = 0; j < BLOCK_SIZE; ++j) off_diagonal[j + j * BLOCK_SIZE] = -1.0;
        if (tinyopt_block_tridiag_set_block(hessian, i, i + 1, off_diagonal) != 0) return 1;
      }
    }
  }
  return 0;
}

int main(void) {
  double x[DIMS] = {0};
  double target[DIMS];
  for (int i = 0; i < DIMS; ++i) target[i] = 0.25 * i - 0.5;

  tinyopt_params_t params = {x, DIMS, NULL};
  tinyopt_block_tridiag_problem_t problem = {NUM_BLOCKS, BLOCK_SIZE, accumulate, target};
  tinyopt_options_t options;
  if (tinyopt_options_default(&options) != TINYOPT_STATUS_OK) return 1;
  options.max_iters = 50;
  options.exp_final_cov = 1;
  double covariance[DIMS * DIMS];
  options.final_covariance_buffer = covariance;
  options.final_covariance_capacity = DIMS * DIMS;

  tinyopt_summary_t summary = {0};
  if (tinyopt_optimize_block_tridiag(&params, &problem, &options, &summary) != TINYOPT_STATUS_OK)
    return 1;
  if (summary.final_covariance_dims != DIMS) return 1;
  if (!(covariance[0] > 0.0)) return 1;
  return gradient_norm(x, target) < 1e-6 ? 0 : 1;
}
