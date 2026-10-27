// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tinyopt/traits/matrix_traits.h>
#include <tinyopt/types.h>
#include <Eigen/Core>

namespace tinyopt {

enum class CovariancePattern : uint8_t {
  Auto,  ///< Natural structure (Dense for dense, BlockTridiagonal for tridiagonal, Schur for
         ///< partitioned)
  BlockDiagonal,     ///< Marginal block diagonal only (Σ_ii)
  PreserveSparsity,  ///< Covariance entries matching the sparsity pattern of H
  SchurPartitioned,  ///< Dense reduced system (e.g. cameras) + block-diagonal eliminated (points)
  FullDense          ///< Full dense matrix (caution: high memory usage on large systems)
};

struct CovarianceOptions {
  CovariancePattern pattern = CovariancePattern::Auto;
  /// If true, scales (J^T J)^-1 by s^2 = chi^2 / (N_res - N_params).
  bool scale_by_residual_variance = false;
  /// Optional ridge regularization epsilon: inverts (H + epsilon * I) to stabilize ill-conditioned
  /// systems.
  double damping = 0.0;
  /// If true, zeroes out rows/cols corresponding to locked/fixed parameters so their variance is
  /// 0.0.
  bool zero_locked_variances = true;
};

/**
 * @brief Structured, lazy covariance wrapper for Schur-partitioned problems (e.g., Bundle
 * Adjustment).
 */
template <typename Scalar, typename SolverType, typename BlockMatrixType>
class SchurCovariance {
 public:
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  SchurCovariance(const SolverType& solver, BlockMatrixType H, Index num_reduced,
                  const CovarianceOptions& options = {}, Scalar scale = Scalar(1))
      : solver_(solver),
        H_(std::move(H)),
        num_reduced_(num_reduced),
        options_(options),
        scale_(scale) {
    const_cast<SolverType&>(solver_).Compute(H_);
  }

  /// Dense covariance of all reduced parameters (e.g. camera poses, size N_r x N_r).
  [[nodiscard]] const DenseMatrix& reduced() const {
    if (!cached_reduced_.has_value()) {
      cached_reduced_ = solver_.ldlt().inverseSchurReduced();
      if (scale_ != Scalar(1)) {
        *cached_reduced_ *= scale_;
      }
    }
    return *cached_reduced_;
  }

  /// Specific reduced sub-block (e.g. camera i, camera j covariance of size Rows x Cols).
  template <int Rows, int Cols>
  [[nodiscard]] Matrix<Scalar, Rows, Cols> reduced_block(Index block_row, Index block_col) const {
    const auto& red = reduced();
    return red.template block<Rows, Cols>(block_row * Rows, block_col * Cols);
  }

  /// Marginal covariance of an eliminated parameter (e.g. 3D point k of size Dim x Dim), computed
  /// in O(k^2).
  template <int Dim>
  [[nodiscard]] Matrix<Scalar, Dim, Dim> eliminated_block(size_t elim_type,
                                                          Index point_index) const {
    const auto& red = reduced();
    Matrix<Scalar, Dim, Dim> blk =
        solver_.ldlt().template marginalEliminatedBlock<Dim>(H_, red, elim_type, point_index);
    if (scale_ != Scalar(1)) {
      blk *= scale_;
    }
    return blk;
  }

  /// Marginal covariance of an eliminated parameter for the default eliminated type (elim_type =
  /// 1).
  template <int Dim>
  [[nodiscard]] Matrix<Scalar, Dim, Dim> eliminated_block(Index point_index) const {
    return eliminated_block<Dim>(1, point_index);
  }

  /// Extract all marginal eliminated blocks sequentially with zero heap reallocations.
  template <int Dim>
  [[nodiscard]] std::vector<Matrix<Scalar, Dim, Dim>> all_eliminated_blocks(
      size_t elim_type = 1) const {
    const Index num_blocks = H_.num_blocks(elim_type);
    std::vector<Matrix<Scalar, Dim, Dim>> blocks;
    blocks.reserve(static_cast<size_t>(num_blocks));
    for (Index i = 0; i < num_blocks; ++i) {
      blocks.push_back(eliminated_block<Dim>(elim_type, i));
    }
    return blocks;
  }

  /// Alias for all_eliminated_blocks.
  template <int Dim>
  [[nodiscard]] std::vector<Matrix<Scalar, Dim, Dim>> eliminated_blocks(
      size_t elim_type = 1) const {
    return all_eliminated_blocks<Dim>(elim_type);
  }

 private:
  const SolverType& solver_;
  BlockMatrixType H_;
  Index num_reduced_ = 0;
  CovarianceOptions options_{};
  Scalar scale_ = Scalar(1);
  mutable std::optional<DenseMatrix> cached_reduced_;
};

}  // namespace tinyopt
