// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
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
  Summary(const Summary&) = default;

  // Move constructor (Crucial!)
  Summary(Summary&&) noexcept = default;

  // Move assignment operator (Crucial!)
  Summary& operator=(Summary&&) noexcept = default;

  // Explicit copy assignment operator using Copy-and-Swap
  Summary& operator=(const Summary& other) {
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

  bool has_final_cov() const { return !std::holds_alternative<std::monostate>(final_cov); }
  MatX final_cov_dense() const {
    return std::visit(
        [](const auto& covariance) -> MatX {
          using CovarianceType = std::decay_t<decltype(covariance)>;
          if constexpr (std::is_same_v<CovarianceType, std::monostate>) {
            return {};
          } else if constexpr (traits::is_block_sparse_matrix_v<CovarianceType>) {
            return covariance.toDense();
          } else {
            return MatX(covariance);
          }
        },
        final_cov);
  }

  SparseMat final_cov_sparse() const {
    return std::visit(
        [](const auto& covariance) -> SparseMat {
          using CovarianceType = std::decay_t<decltype(covariance)>;
          if constexpr (std::is_same_v<CovarianceType, std::monostate>) {
            return {};
          } else if constexpr (traits::is_block_sparse_matrix_v<CovarianceType>) {
            return covariance.toSparse();
          } else if constexpr (traits::is_sparse_matrix_v<CovarianceType>) {
            return covariance;
          } else {
            return covariance.sparseView();
          }
        },
        final_cov);
  }

  const BlockSparseMatrixX<double>* final_cov_block_sparse() const {
    return std::get_if<BlockSparseMatrixX<double>>(&final_cov);
  }

  template <typename CovarianceType>
  void set_final_cov(const CovarianceType& covariance) {
    if constexpr (traits::is_block_sparse_matrix_v<CovarianceType>) {
      final_cov = covariance.toBlockSparseMatX().template cast<double>();
    } else if constexpr (traits::is_sparse_matrix_v<CovarianceType>) {
      final_cov = SparseMat(covariance.template cast<double>());
    } else {
      final_cov = MatX(covariance.template cast<double>());
    }
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

  /// Final covariance matrix (only saved if options.exp.final_cov = true).
  std::variant<std::monostate, MatX, SparseMat, BlockSparseMatrixX<double>> final_cov;

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
