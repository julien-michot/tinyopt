// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include <tinyopt/cost.h>
#include <tinyopt/math.h>  // Defines Matrix and Vector
#include <tinyopt/stop_reasons.h>
#include <tinyopt/time.h>
#include <tinyopt/traits.h>  // Defines parameters_size_v

namespace tinyopt {

/***
 *  @brief Struct containing optimization results
 *
 ***/
struct Output {
  using Scalar = double;

  /// Returns true if the stop reason is not a failure to solve or NaNs or missing residuals
  bool Succeeded() const { return stop_reason >= StopReason::kNone; }

  /// Returns true if the optimization reached the specified minimal error, delta or gradient
  bool Converged() const {
    return stop_reason >= StopReason::kMinError && stop_reason < StopReason::kMaxIters;
  }
  /// @brief Computes an approximation of the covariance matrix.
  ///
  /// This method calculates the covariance matrix, which is an approximation
  /// derived from the inverse of the Hessian matrix (H).
  ///
  /// It is only meaningful at convergence, right Johan?
  ///
  /// @param rescaled (optional) If true, the covariance matrix is rescaled by
  ///                 ε² / (#ε - dims), where ε² is the sum of squared residuals
  ///                 (final_cost²), #ε is the number of residuals (num_residuals),
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
  ///   - ε (final_cost) is the sum of residuals.
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
  template <typename M = MatX>
  std::optional<M> Covariance(bool rescaled = false) const {
    using M2 = std::conditional_t<traits::is_sparse_matrix_v<M>, SparseMat, MatX>; // TODO better way?
    if (!std::holds_alternative<M2>(final_hessian)) return {};
    const auto &final_hessian_dense = std::get<M2>(final_hessian);
    const auto &cov = InvCov(final_hessian_dense);
    if (cov) {
      if (rescaled && num_residuals > final_hessian_dense.cols())
        return cov.value() * (final_cost * final_cost / (num_residuals - final_hessian_dense.cols()));
      else
        return cov.value();
    }
    return {};
  }

  bool has_final_hessian() const {
    return !std::holds_alternative<std::monostate>(final_hessian);
  }
  const auto &final_hessian_dense() const {
    return std::get<MatX>(final_hessian);
  }
  const auto &final_hessian_sparse() const {
    return std::get<SparseMat>(final_hessian);
  }

  /// Last valid error
  Cost final_cost = Cost(std::numeric_limits<Scalar>::max(), 0);
  Scalar final_rerr_dec = std::numeric_limits<Scalar>::max();

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

  /// Final H, excluding any damping (only saved if options.hessian.save = true)
  std::variant<std::monostate, MatX, SparseMat> final_hessian;

  /// True if numerical derivatives were used
  bool num_diff_used = false;

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
