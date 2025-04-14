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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include <tinyopt/math.h>  // Defines Matrix and Vector
#include <tinyopt/stop_reasons.h>
#include <tinyopt/time.h>
#include <tinyopt/traits.h>  // Defines parameters_size_v

namespace tinyopt {

/***
 *  @brief Struct containing optimization results
 *
 ***/
template <typename _H_t = std::nullptr_t>
struct Output {
  using H_t = _H_t;
  using Scalar = std::conditional_t<traits::is_nullptr_v<H_t>, double,
                                    typename traits::params_trait<H_t>::Scalar>;
  using Dx_t = std::conditional_t<traits::is_nullptr_v<H_t>, Vector<Scalar, Dynamic>,
                                  Vector<Scalar, SQRT(traits::params_trait<H_t>::Dims)>>;

  /// Returns true if the stop reason is not a failure to solve or NaNs or missing residuals
  bool Succeeded() const { return stop_reason >= StopReason::kNone; }

  /// Returns true if the optimization reached the specified minimal error, delta or gradient
  bool Converged() const {
    return stop_reason == StopReason::kMinError || stop_reason == StopReason::kMinDeltaNorm ||
           stop_reason == StopReason::kMinGradNorm;
  }
  /// @brief Computes an approximation of the covariance matrix.
  ///
  /// This method calculates the covariance matrix, which is an approximation
  /// derived from the inverse of the Hessian matrix (H).
  ///
  /// @param rescaled (optional) If true, the covariance matrix is rescaled by
  ///                 ε² / (#ε - dims), where ε² is the sum of squared residuals
  ///                 (final_err²), #ε is the number of residuals (num_residuals),
  ///                 and dims is the number of parameters (H.cols()).
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
  ///   - If `rescaled` is false:  (H)^-1
  ///   - If `rescaled` is true and `num_residuals > H.cols()`:
  ///     (H)^-1 * (ε² / (#ε - dims))
  ///
  /// Where:
  ///   - H is the approximated Hessian matrix.
  ///   - ε (final_err) is the sum of residuals.
  ///   - #ε (num_residuals) is the number of residuals.
  ///   - dims (H.cols()) is the number of parameters.
  ///
  /// The rescaling is applied only when the number of residuals exceeds the number of
  /// parameters, indicating an overdetermined system. This rescaling provides a more
  /// statistically sound estimate of the covariance in such scenarios, especially
  /// when noise modeling was not explicitly included in the observations.
  ///
  /// @note
  /// The function relies on the `InvCov(H)` method to compute the inverse of H.
  /// If `InvCov` returns `std::nullopt`, indicating that H is not invertible, this
  /// method also returns `std::nullopt`.
  ///
  /// @tparam H_t The type of the covariance matrix.
  template <typename H = H_t, std::enable_if_t<!std::is_null_pointer_v<H>, int> = 0>
  std::optional<H_t> Covariance(bool rescaled = false) const {
    const auto cov = InvCov(final_H);
    if (!cov) return std::nullopt;  // Covariance can't be estimated
    if (rescaled && num_residuals > final_H.cols()) {
      return cov.value() * (final_err * final_err / (num_residuals - final_H.cols()));
    } else {
      return cov.value();
    }
  }

  /// Last valid error
  Scalar final_err = std::numeric_limits<Scalar>::max();
  Scalar final_rel_err = std::numeric_limits<Scalar>::max();

  /// Stop reason
  StopReason stop_reason = StopReason::kNone;

  /**
   * @name Statistics
   * @{
   */
  uint16_t num_residuals = 0;  ///< Final number of residuals
  uint16_t num_iters = 0;      ///< Final number of iterations
  uint8_t num_failures = 0;    ///< Final number of failures to decrease the error
  uint8_t num_consec_failures =
      0;  ///< Final number of the last consecutive failures to decrease the error

  TimePoint start_time =
      TimePoint::min();   ///< Starting time of the optimization or
                          ///< std::chrono::system_clock::time_point::min() if not started
  float duration_ms = 0;  ///< Cumulated optimization duration

  H_t final_H;  ///< Final H, excluding any damping (only saved if options.save.H = true)

  /** @} */

  /**
   * @name Per iteration results
   * @{
   */

  std::vector<Scalar> errs;     ///< Mean squared accumulated errors of all iterations
  std::vector<Scalar> deltas2;  ///< Squared Step sizes of all iterations
  std::vector<bool> successes;  ///< Step acceptation status for all iterations

  /** @} */
};

}  // namespace tinyopt
