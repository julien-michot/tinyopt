// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>

#include <tinyopt/log.h>
#include <tinyopt/math.h>

namespace tinyopt {

/***
 *  @brief Common Optimization Options
 *
 ***/
struct Options {
  /**
   * @name Solver Type
   * @{
   */
  enum Solver {
    LevenbergMarquardt = 0,
    GaussNewton,
    GradientDescent,
    ConjugateGradient,
    DogLeg,
    BFGS,
    LBFGS,
  };
  /// Which solver to use. Default is LevenbergMarquardt for NLLS problems.
  Solver solver_type = Solver::LevenbergMarquardt;
  /** @} */

  /// Linear system method. LDLT is enabled by default; methods require their CMake option.
  LinearSolverMethod linear_solver = LinearSolverMethod::LDLT;
  /// Relative singular-value cutoff for TruncatedSVD; 0 uses Eigen's default threshold.
  double svd_relative_threshold = 0.0;

  Options(Solver type = Solver::LevenbergMarquardt) : solver_type(type) {};

  /**
   * @name Optimization options
   * @{
   */
  struct Optimization {
    /// Recompute the final cost as a rollback safety check.
    bool check_final_cost = false;

    /// Use relative error decrease as step quality; otherwise use 0.0.
    bool use_step_quality_approx = false;

    /// Gradient clipping to range [-v, +v], disabled if 0.
    float grad_clipping = 0;
  } opt;
  /** @} */

  /**
   * @name Hessian Properties
   * @{
   */

  struct Hessian {
    bool H_is_full = true;  ///< Specify if H is only Upper triangularly or fully filled

    float check_min_H_diag = 0;  ///< Check the the hessian's diagonal are not all below the
    ///< threshold. Use 0 to disable the check.

    bool save_last = true;  ///< Saves the last Hessian `H` as part of the output results
  } hessian;

  /** @} */

  /**
   * @name Cost scaling options (mostly for NLLS solvers really)
   * @{
   */
  struct CostScaling {
    bool use_squared_norm = true;  ///< Use squared norm instead of norm (faster)
    bool downscale_by_2 = false;   ///< Rescale the cost by 0.5
    /// Normalize the final error by the number of residuals (after use_squared_norm)
    bool normalize = false;
  } cost;

  /** @} */

  /**
   * @name Stop criteria
   * @{
   */
  struct StopCriteria {
    uint16_t max_iters = 50;          ///< Maximum number of outer iterations
    float min_error = 1e-12f;         ///< Minimum error/cost
    float min_rerr_dec = 1e-10f;      ///< Minimum relative error decrease
    float min_step_norm2 = 1e-14f;    ///< Minimum squared step norm
    float min_grad_norm2 = 1e-18f;    ///< Minimum squared gradient norm
    uint8_t max_total_failures = 0;   ///< Overall max failures to decrease error
    uint8_t max_consec_failures = 5;  ///< Maximum consecutive failures to decrease error
    double max_duration_ms = 0;       ///< Maximum optimization duration in milliseconds

    std::function<bool(double, double, double)> stop_callback;
    std::function<bool(float, const VecXf &, const VecXf &)> stop_callback2;
  } stop;
  /** @} */

  /**
   * @name Logging Options
   * @{
   */
  struct LogOptions {
    bool enable = true;            ///< Whether to enable the logging
    std::string e = "ε²";          ///< Symbol used when logging the error, e.g ε, ε² or √ε etc.
    bool print_emoji = true;       ///< Whether to show the emoji or not
    bool print_x = false;          ///< Log the value of 'x'
    bool print_dx = false;         ///< Log the value of step 'dx'
    bool print_inliers = false;    ///< Log the inliers ratio (in %)
    bool print_t = true;           ///< Log the duration (in ms)
    bool print_J_jet = false;      ///< Log the value of 'J' from the Jet
    bool print_max_stdev = false;  ///< Log the maximum of all standard deviations
                                   ///< (sqrt((co-)variance)) (need to invert H)
    bool print_failure = false;    // Log when a failure to solve the linear system happens
  } log;
  /** @} */

  struct LM {
    bool jacobi_scaling = false;  ///< Scale normal equations using the clamped Hessian diagonal
    /**
     * @name Damping options
     * @{
     */
    float damping_init = 1e-4f;  ///< Initial damping factor. If 0, the damping is disable (it will
    ///< behave like Gauss-Newton)
    ///< Min and max damping values (only used when damping_init != 0)
    std::array<float, 2> damping_range{{1e-9f, 1e9f}};

    float good_factor = 1.0f / 3.0f;  ///< Scale to apply to the damping for good steps
    float bad_factor = 2.0f;          ///< Scale to apply to the damping for bad steps
    /** @} */
  } lm;

  /**
   * @name Gradient Descent options
   * @{
   */
  struct GD {
    float lr = 1e-3f;  ///< Initial learning rate
    // TODO float min_lr = 1e-6f;  ///< Minimum learning rate
    // TODO float max_lr = 1e6f;   ///< Maximum learning rate
    // TODO float decay_factor = 0.5f;        ///< Factor to decay the learning rate (if adaptive)
    // TODO bool use_adaptive_lr = false;     ///< Whether to use adaptive learning rate
    /** @} */
  } gd;

  struct CG {
    float step_size = 0.25f;
    float step_reduction = 0.5f;
  } cg;

  struct DL {
    float radius_init = 1.0f;
    float radius_max = 1e6f;
    float shrink_factor = 0.25f;
    float expand_factor = 2.0f;
  } dl;

  struct BFGS {
    float step_size = 1.0f;
    float step_reduction = 0.5f;
    float step_growth = 1.5f;
    float max_step_size = 1.0f;
    float curvature_threshold = 1e-8f;
  } bfgs;

  struct LBFGS {
    float step_size = 1.0f;
    float step_reduction = 0.5f;
    float step_growth = 1.5f;
    float max_step_size = 1.0f;
    float curvature_threshold = 1e-8f;
    uint8_t history_size = 8;
  } lbfgs;
};

}  // namespace tinyopt
