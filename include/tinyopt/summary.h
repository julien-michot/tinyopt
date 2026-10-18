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
#include <tinyopt/math/matrix_conversion.h>
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

  /// @brief Computes the covariance matrix based on last Hessian (real or approximated).
  ///
  /// Return std::nullopt if the Hessian is not invertible or missing.
  template <typename M = MatX>
  std::optional<M> Covariance() const {
    return CovarianceImpl<M>(Scalar(1));
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
  /// The function relies on `Inverse(H)` to compute the inverse of H.
  /// If `Inverse` returns `std::nullopt`, indicating that H is not invertible, this
  /// method also returns `std::nullopt`.
  ///
  /// @tparam H_t The type of the covariance matrix.
  template <typename M = MatX>
  std::optional<M> CovarianceRescaled() const {
    const Index dims = final_hessian_cols();
    if (dims == 0 || num_residuals <= dims) return {};
    return CovarianceImpl<M>(final_cost.cost / (num_residuals - dims));
  }

  bool has_final_hessian() const { return !std::holds_alternative<std::monostate>(final_hessian); }
  MatX final_hessian_dense() const {
    return std::visit(
        [](const auto& hessian) -> MatX {
          using HessianType = std::decay_t<decltype(hessian)>;
          if constexpr (std::is_same_v<HessianType, std::monostate>) {
            return {};
          } else if constexpr (traits::is_block_sparse_matrix_v<HessianType>) {
            return hessian.toDense();
          } else {
            return MatX(hessian);
          }
        },
        final_hessian);
  }

  SparseMat final_hessian_sparse() const {
    return std::visit(
        [](const auto& hessian) -> SparseMat {
          using HessianType = std::decay_t<decltype(hessian)>;
          if constexpr (std::is_same_v<HessianType, std::monostate>) {
            return {};
          } else if constexpr (traits::is_block_sparse_matrix_v<HessianType>) {
            return hessian.toSparse();
          } else if constexpr (traits::is_sparse_matrix_v<HessianType>) {
            return hessian;
          } else {
            return hessian.sparseView();
          }
        },
        final_hessian);
  }

  const BlockSparseMatrixX<double>* final_hessian_block_sparse() const {
    return std::get_if<BlockSparseMatrixX<double>>(&final_hessian);
  }

  template <typename HessianType>
  void set_final_hessian(const HessianType& hessian) {
    if constexpr (traits::is_block_sparse_matrix_v<HessianType>) {
      final_hessian = hessian.toBlockSparseMatX().template cast<double>();
    } else if constexpr (traits::is_sparse_matrix_v<HessianType>) {
      final_hessian = SparseMat(hessian.template cast<double>());
    } else {
      final_hessian = MatX(hessian.template cast<double>());
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

  /// Final H, excluding any damping (only saved if options.hessian.save = true)
  std::variant<std::monostate, MatX, SparseMat, BlockSparseMatrixX<double>> final_hessian;

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

 private:
  template <typename M>
  std::optional<M> CovarianceImpl(Scalar scale) const {
    return std::visit(
        [&](const auto& hessian) -> std::optional<M> {
          using HessianType = std::decay_t<decltype(hessian)>;
          if constexpr (std::is_same_v<HessianType, std::monostate>) {
            return {};
          } else if constexpr (traits::is_block_sparse_matrix_v<HessianType>) {
            const auto source = [&] {
              if constexpr (std::is_same_v<HessianType, BlockSparseMatrixX<double>>) {
                return hessian;
              } else {
                return hessian.toBlockSparseMatX().template cast<double>();
              }
            }();
            const auto inverse = Inverse(source.toSparse());
            if (!inverse) return {};
            return internal::ConvertMatrix<M>(*inverse, scale, source.block_sizes(),
                                              source.block_counts());
          } else {
            const auto inverse = Inverse(hessian);
            if (!inverse) return {};
            return internal::ConvertMatrix<M>(*inverse, scale);
          }
        },
        final_hessian);
  }

  Index final_hessian_cols() const {
    return std::visit(
        [](const auto& hessian) -> Index {
          using HessianType = std::decay_t<decltype(hessian)>;
          if constexpr (std::is_same_v<HessianType, std::monostate>) {
            return 0;
          } else {
            return hessian.cols();
          }
        },
        final_hessian);
  }
};

}  // namespace tinyopt
