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

typedef struct tinyopt_summary_t {
  int stop_reason;                    /* Tinyopt termination reason value. */
  int num_iters;                      /* Number of optimizer iterations performed. */
  int num_failures;                   /* Number of unsuccessful trial steps. */
  int num_residuals;                  /* Number of residuals used by the problem. */
  double final_cost;                  /* Final objective cost. */
  int used_numerical_differentiation; /* Nonzero if numerical derivatives were used. */
} tinyopt_summary_t;

typedef enum tinyopt_status_t {
  TINYOPT_STATUS_OK = 0,
  TINYOPT_STATUS_INVALID_ARGUMENT = 1,
  TINYOPT_STATUS_CALLBACK_FAILED = 2,
  TINYOPT_STATUS_RESIDUAL_CALLBACK_FAILED = TINYOPT_STATUS_CALLBACK_FAILED,
  TINYOPT_STATUS_OPTIMIZATION_FAILED = 3,
  TINYOPT_STATUS_INTERNAL_ERROR = 4,
  TINYOPT_STATUS_USER_STOPPED = 5
} tinyopt_status_t;

typedef enum tinyopt_eval_type_t {
  TINYOPT_EVAL_COST_ONLY = 0,
  TINYOPT_EVAL_RESIDUALS,
  TINYOPT_EVAL_GRADIENT,
  TINYOPT_EVAL_HESSIAN
} tinyopt_eval_type_t;

typedef enum tinyopt_solver_t {
  TINYOPT_SOLVER_LEVENBERG_MARQUARDT = 0,
  TINYOPT_SOLVER_GAUSS_NEWTON,
  TINYOPT_SOLVER_GRADIENT_DESCENT,
  TINYOPT_SOLVER_CONJUGATE_GRADIENT,
  TINYOPT_SOLVER_DOGLEG,
  TINYOPT_SOLVER_BFGS,
  TINYOPT_SOLVER_LBFGS
} tinyopt_solver_t;

typedef enum tinyopt_linear_solver_t {
  TINYOPT_LINEAR_SOLVER_LDLT = 0,
  TINYOPT_LINEAR_SOLVER_LLT,
  TINYOPT_LINEAR_SOLVER_LU,
  TINYOPT_LINEAR_SOLVER_QR,
  TINYOPT_LINEAR_SOLVER_SVD,
  TINYOPT_LINEAR_SOLVER_SUITESPARSE
} tinyopt_linear_solver_t;

typedef int (*tinyopt_stop_callback_t)(double error, double step_norm_squared,
                                       double gradient_norm_squared, void *user_data);
typedef int (*tinyopt_stop_callback2_t)(float error, const float *x, const float *dx, int dims,
                                        void *user_data);
/* Called after every parameter update with the step `dx` added to the parameters. `is_rollback` is
   nonzero when a rejected step is undone, `dx` being then the negated step. Summing the steps of
   all calls tracks the current parameters. Return nonzero to stop. */
typedef int (*tinyopt_step_callback_t)(const float *dx, int dims, int is_rollback, void *user_data);

typedef struct tinyopt_options_t {
  tinyopt_solver_t solver_type;          /* Nonlinear optimizer to use. */
  tinyopt_linear_solver_t linear_solver; /* Linear system solver to use. */
  double svd_relative_threshold;         /* Relative singular-value cutoff for SVD solvers. */

  int check_final_cost;        /* Re-evaluate and validate the final cost. */
  int use_step_quality_approx; /* Approximate step quality to reduce cost evaluations. */
  float grad_clipping;         /* Maximum gradient component magnitude; zero disables clipping. */

  int hessian_is_full;              /* Treat the accumulated Hessian as a full matrix. */
  float check_min_hessian_diagonal; /* Minimum accepted Hessian diagonal value. */
  int save_last_hessian;            /* Retain the final Hessian in the optimization summary. */

  int use_squared_norm;      /* Use squared residual norm for the objective. */
  int downscale_cost_by_two; /* Multiply the cost by one half. */
  int normalize_cost;        /* Normalize cost by the number of residuals. */

  unsigned short max_iters;          /* Maximum number of optimizer iterations. */
  float min_error;                   /* Stop when the objective error is below this threshold. */
  float min_relative_error_decrease; /* Stop when relative cost decrease is below this value. */
  float min_step_norm_squared;       /* Stop when squared step norm is below this value. */
  float min_gradient_norm_squared;   /* Stop when squared gradient norm is below this value. */
  unsigned char max_total_failures;  /* Maximum total unsuccessful trial steps. */
  unsigned char max_consecutive_failures; /* Maximum consecutive unsuccessful trial steps. */
  double max_duration_ms; /* Maximum optimization duration in milliseconds; zero disables limit. */
  tinyopt_stop_callback_t stop_callback;   /* Optional per-iteration scalar stop callback. */
  void *stop_callback_user_data;           /* User data passed to stop_callback. */
  tinyopt_stop_callback2_t stop_callback2; /* Optional per-iteration vector stop callback. */
  void *stop_callback2_user_data;          /* User data passed to stop_callback2. */
  tinyopt_step_callback_t step_callback;   /* Optional callback run after every parameter update. */
  void *step_callback_user_data;           /* User data passed to step_callback. */

  int log_enabled;                      /* Enable optimizer logging. */
  const char *log_error_symbol;         /* Optional symbol used to label the logged error. */
  int log_print_emoji;                  /* Include status symbols in log output. */
  int log_print_x;                      /* Print parameter values. */
  int log_print_dx;                     /* Print parameter updates. */
  int log_print_inliers;                /* Print inlier statistics when available. */
  int log_print_time;                   /* Print iteration timing. */
  int log_print_jacobian_jet;           /* Print Jacobian automatic-differentiation details. */
  int log_print_max_standard_deviation; /* Print maximum parameter standard deviation. */
  int log_print_failure;                /* Print details for unsuccessful trial steps. */

  int lm_jacobi_scaling; /* Scale LM damping by the Hessian diagonal. */
  float lm_damping_init; /* Initial Levenberg-Marquardt damping. */
  float lm_damping_min;  /* Minimum Levenberg-Marquardt damping. */
  float lm_damping_max;  /* Maximum Levenberg-Marquardt damping. */
  float lm_good_factor;  /* Damping multiplier after a successful step. */
  float lm_bad_factor;   /* Damping multiplier after an unsuccessful step. */

  float gd_learning_rate;           /* Gradient-descent learning rate. */
  float cg_step_size;               /* Initial conjugate-gradient step size. */
  float cg_step_reduction;          /* Conjugate-gradient step reduction factor. */
  float dogleg_radius_init;         /* Initial Dogleg trust-region radius. */
  float dogleg_radius_max;          /* Maximum Dogleg trust-region radius. */
  float dogleg_shrink_factor;       /* Dogleg radius multiplier after an unsuccessful step. */
  float dogleg_expand_factor;       /* Dogleg radius multiplier after a successful step. */
  float bfgs_step_size;             /* Initial BFGS step size. */
  float bfgs_step_reduction;        /* BFGS step multiplier after an unsuccessful step. */
  float bfgs_step_growth;           /* BFGS step multiplier after a successful step. */
  float bfgs_max_step_size;         /* Maximum BFGS step size. */
  float bfgs_curvature_threshold;   /* Minimum curvature for a BFGS update. */
  float lbfgs_step_size;            /* Initial L-BFGS step size. */
  float lbfgs_step_reduction;       /* L-BFGS step multiplier after an unsuccessful step. */
  float lbfgs_step_growth;          /* L-BFGS step multiplier after a successful step. */
  float lbfgs_max_step_size;        /* Maximum L-BFGS step size. */
  float lbfgs_curvature_threshold;  /* Minimum curvature for an L-BFGS update. */
  unsigned char lbfgs_history_size; /* Number of past updates retained by L-BFGS. */
} tinyopt_options_t;

TINYOPT_C_API tinyopt_status_t tinyopt_options_default(tinyopt_options_t *options);

#ifdef __cplusplus
}
#endif

#endif