// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef TINYOPT_C_C_API_COMMON_H
#define TINYOPT_C_C_API_COMMON_H

#include <tinyopt/c/c_api_config.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !TINYOPT_C_API_ENABLE_SHARED
#define TINYOPT_C_API
#elif defined(_WIN32)
#if defined(TINYOPT_C_BUILD)
#define TINYOPT_C_API __declspec(dllexport)
#else
#define TINYOPT_C_API __declspec(dllimport)
#endif
#else
#define TINYOPT_C_API __attribute__((visibility("default")))
#endif

typedef struct tinyopt_summary {
  int stop_reason;
  int num_iters;
  int num_failures;
  int num_residuals;
  double final_cost;
  int used_numerical_differentiation;
} tinyopt_summary;

typedef enum tinyopt_status {
  TINYOPT_STATUS_OK = 0,
  TINYOPT_STATUS_INVALID_ARGUMENT = 1,
  TINYOPT_STATUS_CALLBACK_FAILED = 2,
  TINYOPT_STATUS_RESIDUAL_CALLBACK_FAILED = TINYOPT_STATUS_CALLBACK_FAILED,
  TINYOPT_STATUS_OPTIMIZATION_FAILED = 3,
  TINYOPT_STATUS_INTERNAL_ERROR = 4,
  TINYOPT_STATUS_USER_STOPPED = 5
} tinyopt_status;

typedef enum tinyopt_eval_type {
  TINYOPT_EVAL_COST_ONLY = 0,
  TINYOPT_EVAL_RESIDUALS,
  TINYOPT_EVAL_GRADIENT,
  TINYOPT_EVAL_HESSIAN
} tinyopt_eval_type;

typedef enum tinyopt_solver {
  TINYOPT_SOLVER_LEVENBERG_MARQUARDT = 0,
  TINYOPT_SOLVER_GAUSS_NEWTON,
  TINYOPT_SOLVER_GRADIENT_DESCENT,
  TINYOPT_SOLVER_CONJUGATE_GRADIENT,
  TINYOPT_SOLVER_DOGLEG,
  TINYOPT_SOLVER_BFGS,
  TINYOPT_SOLVER_LBFGS
} tinyopt_solver;

typedef enum tinyopt_linear_solver {
  TINYOPT_LINEAR_SOLVER_LDLT = 0,
  TINYOPT_LINEAR_SOLVER_LLT,
  TINYOPT_LINEAR_SOLVER_LU,
  TINYOPT_LINEAR_SOLVER_QR,
  TINYOPT_LINEAR_SOLVER_SVD,
  TINYOPT_LINEAR_SOLVER_SUITESPARSE,
  TINYOPT_LINEAR_SOLVER_TRUNCATED_SVD
} tinyopt_linear_solver;

typedef int (*tinyopt_stop_callback)(double error, double step_norm_squared,
                                     double gradient_norm_squared, void *user_data);
typedef int (*tinyopt_stop_callback2)(float error, const float *x, const float *dx, int dims,
                                      void *user_data);

typedef struct tinyopt_options {
  tinyopt_solver solver_type;
  tinyopt_linear_solver linear_solver;
  double svd_relative_threshold;

  int check_final_cost;
  int use_step_quality_approx;
  float grad_clipping;

  int hessian_is_full;
  float check_min_hessian_diagonal;
  int save_last_hessian;

  int use_squared_norm;
  int downscale_cost_by_two;
  int normalize_cost;

  unsigned short max_iters;
  float min_error;
  float min_relative_error_decrease;
  float min_step_norm_squared;
  float min_gradient_norm_squared;
  unsigned char max_total_failures;
  unsigned char max_consecutive_failures;
  double max_duration_ms;
  tinyopt_stop_callback stop_callback;
  void *stop_callback_user_data;
  tinyopt_stop_callback2 stop_callback2;
  void *stop_callback2_user_data;

  int log_enabled;
  const char *log_error_symbol;
  int log_print_emoji;
  int log_print_x;
  int log_print_dx;
  int log_print_inliers;
  int log_print_time;
  int log_print_jacobian_jet;
  int log_print_max_standard_deviation;
  int log_print_failure;

  int lm_jacobi_scaling;
  float lm_damping_init;
  float lm_damping_min;
  float lm_damping_max;
  float lm_good_factor;
  float lm_bad_factor;

  float gd_learning_rate;
  float cg_step_size;
  float cg_step_reduction;
  float dogleg_radius_init;
  float dogleg_radius_max;
  float dogleg_shrink_factor;
  float dogleg_expand_factor;
  float bfgs_step_size;
  float bfgs_step_reduction;
  float bfgs_step_growth;
  float bfgs_max_step_size;
  float bfgs_curvature_threshold;
  float lbfgs_step_size;
  float lbfgs_step_reduction;
  float lbfgs_step_growth;
  float lbfgs_max_step_size;
  float lbfgs_curvature_threshold;
  unsigned char lbfgs_history_size;
} tinyopt_options;

TINYOPT_C_API tinyopt_status tinyopt_options_default(tinyopt_options *options);

#ifdef __cplusplus
}
#endif

#endif