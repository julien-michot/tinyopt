// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
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
struct Summary {
  using Scalar = double;

  // Default constructor
  Summary() = default;

  // Copy constructor
  Summary(const Summary &) = default;

  // Move constructor (Crucial!)
  Summary(Summary &&) noexcept = default;

  // Move assignment operator (Crucial!)
  Summary &operator=(Summary &&) noexcept = default;

  // Explicit copy assignment operator using Copy-and-Swap
  Summary &operator=(const Summary &other) {
    if (this != &other) {
      Summary temp(other);      // Copy constructor copies all fields safely
      *this = std::move(temp);  // Now calls the move assignment operator above!
    }
    return *this;
  }

  /// Returns true if the stop reason is not a failure to solve or NaNs or missing residuals
  bool Succeeded() const { return stop_reason >= StopReason::kNone; }

  /// Returns true if the optimization reached the specified minimal error, delta or gradient
  bool Converged() const {
    return stop_reason >= StopReason::kMinError && stop_reason < StopReason::kMaxIters;
  }

  /// @brief Computes the covariance matrix based on last Hessian (real or approximated).
  ///
  /// Return std::nullopt if the Hessian is not invertible or missing.
  template <typename M = MatX>
  std::optional<M> Covariance() const {
    using M2 =
        std::conditional_t<traits::is_sparse_matrix_v<M>, SparseMat, MatX>;  // TODO better way?
    if (!std::holds_alternative<M2>(final_hessian)) return {};
    const auto &final_hessian_dense = std::get<M2>(final_hessian);
    return InvCov(final_hessian_dense);
  }

  /// @brief Computes the covariance matrix based on last Hessian (real or approximated).
  ///
  /// Return std::nullopt if the Hessian is not invertible or missing.
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
  ///     (H)^-1 * (SSE / (num_residuals - dims))
  ///
  /// Where:
  ///   - H is the approximated Hessian matrix.
  ///   - SSE (`final_cost.cost`) is the sum of squared residuals.
  ///   - `num_residuals` is the number of scalar residuals.
  ///   - dims (H.cols()) is the number of parameters.
  ///
  /// Rescaling requires positive residual degrees of freedom. It is not valid for robust losses,
  /// normalized or downscaled costs, or arbitrary scalar objectives whose cost is not the raw
  /// residual sum of squares. The caller must ensure the residuals meet the statistical
  /// assumptions above.
  ///
  /// @note
  /// The function relies on the `InvCov(H)` method to compute the inverse of H.
  /// If `InvCov` returns `std::nullopt`, indicating that H is not invertible, this
  /// method also returns `std::nullopt`.
  ///
  /// @tparam H_t The type of the covariance matrix.
  template <typename M = MatX>
  std::optional<M> CovarianceRescaled() const {
    using M2 =
        std::conditional_t<traits::is_sparse_matrix_v<M>, SparseMat, MatX>;  // TODO better way?
    if (!std::holds_alternative<M2>(final_hessian)) return {};
    const auto &final_hessian_dense = std::get<M2>(final_hessian);
    const auto &cov = InvCov(final_hessian_dense);
    if (cov) {
      if (num_residuals <= final_hessian_dense.cols()) return {};
      return cov.value() *
             (final_cost.cost / (num_residuals - final_hessian_dense.cols()));
    }
    return {};
  }

  bool has_final_hessian() const { return !std::holds_alternative<std::monostate>(final_hessian); }
  const auto &final_hessian_dense() const { return std::get<MatX>(final_hessian); }
  const auto &final_hessian_sparse() const { return std::get<SparseMat>(final_hessian); }

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

  struct History {
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
    std::array<Scalar, 2> errs{};
    std::array<Scalar, 2> deltas2{};
    std::array<bool, 2> successes{};
    std::size_t size = 0;

    void Add(Scalar err, Scalar delta2, bool success) {
      const std::size_t index = size == 0 ? 0 : 1;
      errs[index] = err;
      deltas2[index] = delta2;
      successes[index] = success;
      if (size < 2) ++size;
    }

    bool empty() const { return size == 0; }
    Scalar last_delta2() const { return deltas2[size == 1 ? 0 : 1]; }
#else
    std::vector<Scalar> errs;     ///< Mean squared accumulated errors of all iterations
    std::vector<Scalar> deltas2;  ///< Squared step sizes of all iterations
    std::vector<bool> successes;  ///< Step acceptance status

    void Add(Scalar err, Scalar delta2, bool success) {
      errs.emplace_back(err);
      deltas2.emplace_back(delta2);
      successes.emplace_back(success);
    }

    bool empty() const { return deltas2.empty(); }
    Scalar last_delta2() const { return deltas2.back(); }
#endif
  } hist;

  /** @} */
};

}  // namespace tinyopt
