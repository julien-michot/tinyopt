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
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <vector>

#include <Eigen/Core>

#include <tinyopt/math.h>    // Defines Matrix and Vector
#include <tinyopt/traits.h>  // Defines parameters_size_v

namespace tinyopt {

/// @brief The reason why the optimization terminated
enum StopReason : int {
  /**
   * @name Failures (negative enums)
   * @{
   */
  kOutOfMemory = -4,        ///< Out of memory when allocating the system (Hessian(s)
  kSolverFailed = -3,       ///< Failed to solve the normal equations (H is not definite positive)
  kSystemHasNaNOrInf = -2,  ///< Residuals or Jacobians have NaNs or Infinity
  kSkipped = -1,            ///< The system has no residuals or nothing to optimize or H is all 0s
  /** @} */
  /**
   * @name Success (positive enums or 0)
   * @{
   */
  kMaxIters = 0,        ///< Reached maximum number of iterations (success)
  kMinDeltaNorm = 1,    ///< Reached minimal delta norm (success)
  kMinGradNorm = 2,     ///< Reached minimal gradient (success)
  kMaxFails = 3,        ///< Failed to decrease error too many times (success)
  kMaxConsecFails = 4,  ///< Failed to decrease error consecutively too many times (success)
  kTimedOut = 5         ///< Total allocated time reached (success)
  /** @} */
};

/***
 *  @brief Struct containing optimization results
 *
 ***/
template <typename H_t>
struct Output {
  /// Last valid step results
  float last_err2 = std::numeric_limits<float>::max();

  /// Stop reason
  StopReason stop_reason = StopReason::kSkipped;

  /// Stop reason description
  std::string StopReasonDescription() const {
    std::ostringstream os;
    switch (stop_reason) {
      /**
       * @name Successes
       * @{
       */
      case StopReason::kMaxIters:
        os << "Reached maximum number of iterations (success)";
        break;
      case StopReason::kMinDeltaNorm:
        os << "Reached minimal delta norm (success)";
        break;
      case StopReason::kMinGradNorm:
        os << "Reached minimal gradient (success)";
        break;
      case StopReason::kMaxFails:
        os << "Failed to decrease error too many times (success)";
        break;
      case StopReason::kMaxConsecFails:
        os << "Failed to decrease error consecutively too many times (success)";
        break;
      case StopReason::kTimedOut:
        os << "Reached maximum allocated time (success)";
        break;
        /** @} */

        /**
         * @name Failures
         * @{
         */
      case StopReason::kOutOfMemory:
        os << "Out of memory when allocating the Hessian(s), use SparseMatrix? (failure)";
        break;
      case StopReason::kSystemHasNaNOrInf:
        os << "Residuals or Jacobians have NaNs or Inf (failure)";
        break;
      case StopReason::kSolverFailed:
        os << "Failed to solve the normal equations (failure)";
        break;
      case StopReason::kSkipped:
        os << "The system has no residuals or nothing to optimize (failure)";
        break;
        /** @} */
      default:
        os << "Unknown reason:" << (int)stop_reason;
        break;
    }
    return os.str();
  }

  /// Returns true if the stop reason is not a failure to solve or NaNs or missing residuals
  bool Succeeded() const { return stop_reason >= 0; }

  /// Returns true if the optimization reached the specified minimal delta norm or gradient norm
  bool Converged() const {
    return stop_reason == StopReason::kMinDeltaNorm || stop_reason == StopReason::kMinGradNorm;
  }
  /// @brief Computes an approximation of the covariance matrix.
  ///
  /// This method calculates the covariance matrix, which is an approximation
  /// derived from the inverse of the Hessian matrix (H). The Hessian matrix
  /// is approximated as 2 * H, where J is the Jacobian matrix.
  ///
  /// @param rescaled (optional) If true, the covariance matrix is rescaled by
  ///                 ε² / (#ε - dims), where ε² is the sum of squared residuals
  ///                 (last_err2), #ε is the number of residuals (num_residuals),
  ///                 and dims is the number of parameters (last_H.cols()).
  ///                 This rescaling is useful when observations lack explicit
  ///                 noise modeling and provides a more accurate estimate of the
  ///                 covariance. Defaults to false.
  ///
  /// @return std::optional<H_t> Returns the computed covariance matrix if H is
  ///                              invertible, wrapped in a std::optional. Returns
  ///                              std::nullopt if H is not invertible, indicating
  ///                              that the covariance cannot be estimated.
  ///
  /// @details
  /// The covariance matrix is calculated as:
  ///
  ///   - If `rescaled` is false:  0.5 * (H)^-1
  ///   - If `rescaled` is true and `num_residuals > last_H.cols()`:
  ///     (H)^-1 * (ε² / (#ε - dims))
  ///
  /// Where:
  ///   - H is the approximated Hessian matrix.
  ///   - ε² (last_err2) is the sum of squared residuals.
  ///   - #ε (num_residuals) is the number of residuals.
  ///   - dims (last_H.cols()) is the number of parameters.
  ///
  /// The factor of 0.5 in the non-rescaled case arises from the Hessian approximation
  /// being 2 * H.
  ///
  /// The rescaling is applied only when the number of residuals exceeds the number of
  /// parameters, indicating an overdetermined system. This rescaling provides a more
  /// statistically sound estimate of the covariance in such scenarios, especially
  /// when noise modeling was not explicitly included in the observations.
  ///
  /// @note
  /// The function relies on the `InvCov(last_H)` method to compute the inverse of H.
  /// If `InvCov` returns `std::nullopt`, indicating that H is not invertible, this
  /// method also returns `std::nullopt`.
  ///
  /// @tparam H_t The type of the covariance matrix.
  std::optional<H_t> Covariance(bool rescaled = false) const {
    const auto cov = InvCov(last_H);
    if (!cov) return std::nullopt;  // Covariance can't be estimated
    if (rescaled && num_residuals > last_H.cols()) {
      return cov.value() * (last_err2 / (num_residuals - last_H.cols()));
    } else {
      return cov.value();
    }
  }

  /**
   * @name Statistics
   * @{
   */
  uint16_t num_residuals = 0;  ///< Final number of residuals
  uint16_t num_iters = 0;      ///< Final number of iterations
  uint8_t num_failures = 0;    ///< Final number of failures to decrease the error
  uint8_t num_consec_failures =
      0;       ///< Final number of the last consecutive failures to decrease the error
  H_t last_H;  ///< Final H, including damping

  /** @} */

  /**
   * @name Per iteration results
   * @{
   */

  std::vector<float> errs2;     ///< Mean squared accumulated errors of all iterations
  std::vector<float> deltas2;   ///< Step sizes of all iterations
  std::vector<bool> successes;  ///< Step acceptation status for all iterations

  /** @} */
};

}  // namespace tinyopt
