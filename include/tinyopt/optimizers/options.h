// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <functional>
#include "tinyopt/math.h"

#include <tinyopt/log.h>

namespace tinyopt {

/***
 *  @brief Common Optimization Options
 *
 ***/
struct Options1 {
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

  /** @} */

  /**
   * @name Stop criteria
   * @{
   */

  uint16_t max_iters = 50;          ///< Maximum number of outter iterations
  float min_error = 1e-6f;          ///< Minimum error/cost
  float min_rerr_dec = 1e-6f;       ///< Minimum relative error (ε_rel = (ε_prev-ε_new)/ε_prev)
  float min_step_norm2 = 1e-16f;    ///< Minimum step (dx) squared norm
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
  } log;
  /** @} */
};

/***
 *  @brief Common Optimization Options for second order methods
 *
 ***/
struct Options2 : Options1 {
  Options2(const Options1 &options = {}) : Options1{options} {}

  /**
   * @name Export Options
   * @{
   */
  struct Export {
    bool H = true;  ///< Saves the last Hessian `H` as part of the output results
  } save;
  /** @} */
};

}  // namespace tinyopt
