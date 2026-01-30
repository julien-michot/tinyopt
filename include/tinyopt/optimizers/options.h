// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>

#include <tinyopt/math.h>
#include <tinyopt/log.h>

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
  };
  /// Which solver to use. Default is LevenbergMarquardt for NLLS problems.
  Solver solver_type = Solver::LevenbergMarquardt;
  /** @} */

  Options(Solver type = Solver::LevenbergMarquardt) : solver_type(type) {};

  /**
   * @name Optimization options
   * @{
   */

  /// Recompute the current error with latest state to eventually roll back. Only
  /// performed at the very last iteration as a safety measure (to prevent unlucky
  /// divergence at the very end...).
  bool check_final_cost = false;

  /// Use relative error decrease as step quality, other 0.0 will be used
  bool use_step_quality_approx = false;

  /// Gradient clipping to range [-v, +v], disabled if 0
  float grad_clipping = 0;

  /** @} */

    /**
     * @name Hessian Properties
     * @{
     */

  struct Hessian {
    bool use_ldlt = true;   ///< If not, will use H.inverse() without any checks on invertibility
    ///< except for Dims==1
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

  uint16_t max_iters = 50;          ///< Maximum number of outter iterations
  float min_error = 1e-12f;         ///< Minimum error/cost
  float min_rerr_dec = 1e-10f;      ///< Minimum relative error (ε_rel = (ε_prev-ε_new)/ε_prev)
  float min_step_norm2 = 1e-14f;    ///< Minimum step (dx) squared norm
  float min_grad_norm2 = 1e-18f;    ///< Minimum gradient squared norm
  uint8_t max_total_failures = 0;   ///< Overall max failures to decrease error
  uint8_t max_consec_failures = 5;  ///< Maximum consecutive failures to decrease error
  double max_duration_ms = 0;       ///< Maximum optimization duration in milliseconds (ms)

  std::function<bool(double, double, double)>
      stop_callback;  ///< User defined callback. It will be called with the current error, step
                      ///< size and the gradient norm, i.e. stop = stop_callback(ε, |δx|², ∇). The
                      ///< user returns `true` to stop the optimization iterations early.

  std::function<bool(float, const VecXf &, const VecXf &)>
      stop_callback2;  ///< User defined callback. It will be called with the current error, step
                       ///< vector and the gradient, i.e. stop = stop_callback(ε, δx, ∇). The user
                       ///< returns `true` to stop the optimization iterations early.
                       /** @} */

  /**
   * @name Logging Options
   * @{
   */
  struct {
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
    bool print_failure = false;  // Log when a failure to solve the linear system happens
  } log;
  /** @} */

  struct LM {
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
    //TODO float min_lr = 1e-6f;  ///< Minimum learning rate
    //TODO float max_lr = 1e6f;   ///< Maximum learning rate
    //TODO float decay_factor = 0.5f;        ///< Factor to decay the learning rate (if adaptive)
    //TODO bool use_adaptive_lr = false;     ///< Whether to use adaptive learning rate
    /** @} */
    } gd;

};

}  // namespace tinyopt
