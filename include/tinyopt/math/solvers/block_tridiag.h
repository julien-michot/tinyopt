// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tinyopt/math/block_tridiag_ldlt.h>
#include <tinyopt/math/solvers/base.h>
#include <tinyopt/types.h>

namespace tinyopt {

/**
 * @brief High-performance linear step solver for BlockTridiagMatrix systems.
 *
 * Backed by BlockTridiagLDLT, solving Ax = b in O(N) operations without dynamic memory allocations.
 */
template <typename BlockMatrix>
struct BlockTridiagLinearSolvers {
  using Scalar = typename BlockMatrix::ScalarType;
  static constexpr int kBlockSize = BlockMatrix::kBlockSize;
  using LDLTType = BlockTridiagLDLT<Scalar, kBlockSize>;

  Index dims = 0;
  Index num_reduced_ = 0;
  LinearSolverOptions options{};

  explicit BlockTridiagLinearSolvers(const LinearSolverOptions& solver_options = {})
      : options(solver_options) {}

  BlockTridiagLinearSolvers(LinearSolverMethod, Index,
                            const LinearSolverOptions& solver_options = {})
      : options(solver_options) {}

  /// Clears analysis and factorization states.
  void reset() {
    dims = 0;
    num_reduced_ = 0;
    analyzed_ = false;
    computed_ = false;
  }

  /// Resizes the solver dimension.
  void resize(Index new_dims, Index num_reduced = 0) {
    if (dims != new_dims || num_reduced_ != num_reduced) {
      dims = new_dims;
      num_reduced_ = num_reduced;
      reset();
    }
  }

  /// Analyzes the tridiagonal sparsity pattern and prepares factorization buffers.
  bool Analyze(const BlockMatrix& H, Index num_reduced = 0) {
    analyzed_ = false;
    computed_ = false;
    const Index dimension = H.rows();
    if (H.cols() != dimension || dimension <= 0) {
      return false;
    }

    dims = dimension;
    num_reduced_ = num_reduced;

    ldlt_.analyzePattern(H);
    analyzed_ = ldlt_.isAnalyzed();
    return analyzed_;
  }

  /// Numerical factorization: factors the block tridiagonal system.
  bool Compute(const BlockMatrix& H) {
    computed_ = false;
    if (!analyzed_ || H.rows() != dims || H.cols() != dims) {
      return false;
    }
    ldlt_.compute(H);
    computed_ = (ldlt_.info() == Eigen::Success);
    return computed_;
  }

  /// Solves the linear system H * solution = rhs (analyzing and computing if needed).
  std::optional<Vector<Scalar, Dynamic>> Solve(const BlockMatrix& H,
                                               const Vector<Scalar, Dynamic>& rhs) {
    if (!analyzed_) {
      if (!Analyze(H, num_reduced_)) return std::nullopt;
    }
    if (!Compute(H)) return std::nullopt;
    return SolverLinear(H, rhs);
  }

  /// Direct linear solve without re-analyzing or re-computing if already computed.
  std::optional<Vector<Scalar, Dynamic>> SolverLinear(const BlockMatrix&,
                                                      const Vector<Scalar, Dynamic>& rhs) const {
    if (!computed_ || rhs.size() != dims || !rhs.allFinite()) return std::nullopt;
    const Vector<Scalar, Dynamic> solution = ldlt_.solve(rhs);
    if (!solution.allFinite()) return std::nullopt;
    return solution;
  }

  /// Convenience overload taking only the rhs vector.
  std::optional<Vector<Scalar, Dynamic>> SolverLinear(const Vector<Scalar, Dynamic>& rhs) const {
    if (!computed_ || rhs.size() != dims || !rhs.allFinite()) return std::nullopt;
    const Vector<Scalar, Dynamic> solution = ldlt_.solve(rhs);
    if (!solution.allFinite()) return std::nullopt;
    return solution;
  }

  /// Computes the block tridiagonal covariance matrix without matrix conversions or allocations.
  std::optional<BlockMatrix> Inverse(const BlockMatrix& H) {
    if (!analyzed_) {
      if (!Analyze(H, num_reduced_)) return std::nullopt;
    }
    if (!Compute(H)) return std::nullopt;
    return ldlt_.inverseBlockTridiag();
  }

  /// Single block covariance query (i, j).
  std::optional<typename BlockMatrix::Block> BlockCovariance(Index i, Index j) const {
    if (!computed_) return std::nullopt;
    return ldlt_.blockCovariance(i, j);
  }

  /// Single block covariance query (i, j) taking matrix.
  std::optional<typename BlockMatrix::Block> BlockCovariance(const BlockMatrix& H, Index i,
                                                             Index j) {
    if (!analyzed_) {
      if (!Analyze(H, num_reduced_)) return std::nullopt;
    }
    if (!Compute(H)) return std::nullopt;
    return ldlt_.blockCovariance(i, j);
  }

  [[nodiscard]] const LDLTType& ldlt() const noexcept { return ldlt_; }
  [[nodiscard]] bool IsAnalyzed() const noexcept { return analyzed_; }
  [[nodiscard]] bool IsComputed() const noexcept { return computed_; }
  [[nodiscard]] LinearSolverMethod GetMethod() const { return options.method; }
  [[nodiscard]] Index Dims() const { return dims; }

 private:
  LDLTType ldlt_;
  bool analyzed_ = false;
  bool computed_ = false;
};

}  // namespace tinyopt
