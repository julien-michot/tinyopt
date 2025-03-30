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

#include <tinyopt/lm.h>

namespace tinyopt::gn {

/***
 *  @brief GN Optimization options
 *
 ***/
struct Options : tinyopt::CommonOptions {
  Options(const tinyopt::CommonOptions &options = {}) : tinyopt::CommonOptions(options) {
    // No damping so we can stop at first failure to descrease the error
    this->max_total_failures = 1;
    this->max_consec_failures = 1;
  }
};

/**
 * @brief Minimize a loss function using the Gauss-Newton method.
 *
 * This function optimizes a set of parameters `x` to minimize a given loss function,
 * employing the Gauss-Newton minimization algorithm.
 *
 * @tparam ParametersType Type of the parameters to be optimized. Must support arithmetic operations
 * and assignment.
 * @tparam ResidualsFunc Type of the residuals function. Must be callable with ParametersType and
 * return a scalar or a vector of residuals. The function signature is either f(x) or f(x, JtJ,
 * Jt_res).
 *
 * @param[in,out] x The initial and optimized parameters. Modified in-place.
 * @param[in] func The residual function to be minimized. It should return a vector of residuals
 * based on the input parameters.
 * @param[in] options Optional parameters for the optimization process (e.g., tolerances, max
 * iterations). Defaults to `Options{}`.
 *
 * @return The optimization details (`Output` struct).
 *
 * @code
 * // Example usage:
 * float x = 1;
 * const auto &out = Optimize(x, [](const auto &x) { return x * x - 2.0; });
 * @endcode
 */
template <typename ParametersType, typename ResidualsFunc>
inline auto Optimize(ParametersType &x, const ResidualsFunc &func,
                     const Options &_options = Options{}) {
  lm::Options options{_options};
  options.damping_init = 0; // Disable damping
  if constexpr (std::is_invocable_v<ResidualsFunc, const ParametersType &>) {
    const auto optimize = [](auto &x, const auto &func, const auto &options) {
      return lm::LM(x, func, options);
    };
    return OptimizeJet(x, func, optimize, options);
  } else {
    return lm::LM(x, func, options);
  }
}
}  // namespace tinyopt::gn
