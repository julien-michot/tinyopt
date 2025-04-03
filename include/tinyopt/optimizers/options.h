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

#include <tinyopt/log.h>

namespace tinyopt {

/***
 *  @brief Common Optimization Options
 *
 ***/
struct Options1 {
  /**
   * @name Stop criteria
   * @{
   */

  uint16_t num_iters = 100;         ///< Maximum number of outter iterations
  float min_error = 0;              ///< Minimum error
  float min_delta_norm2 = 0;        ///< Minimum delta (step) squared norm
  float min_grad_norm2 = 1e-12;     ///< Minimum gradient squared norm
  uint8_t max_total_failures = 0;   ///< Overall max failures to decrease error
  uint8_t max_consec_failures = 3;  ///< Maximum consecutive failures to decrease error
  double max_duration_ms = 0;       ///< Maximum optimization duration in milliseconds (ms)

  std::function<bool(double, double, double)>
      stop_callback;  ///< User defined callback, will be called with the current error and
                      ///< the gradient norm, i.e. stop = stop_callback(ε, |δX|², ∇ε). The user
                      ///< returns `true` to stop the optimization iterations early.

  /** @} */

  /**
   * @name Logging Options
   * @{
   */
  struct {
    bool enable = true;            ///< Whether to enable the logging
    bool print_x = true;           ///< Log the value of 'x'
    bool print_t = true;           ///< Log the duration (in ms)
    bool print_mean_x = false;     ///< Log the Mean Error (ε/#ε) instead of ε
    bool print_J_jet = false;      ///< Log the value of 'J' from the Jet
    bool print_failure = true;     ///< Log the value of 'H' and 'grad' from the Jet
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

  bool export_H = true;  ///< Save and return the last H as part of the output
                         ///< (only for NLLS & not 1st degree methods)

  /** @} */
};

}  // namespace tinyopt
