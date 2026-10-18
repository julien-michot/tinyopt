// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tinyopt/math/block_sparse_ldlt.h>
#include <tinyopt/math/solvers/base.h>
#include <tinyopt/types.h>

namespace tinyopt {

/**
 * @brief High-performance linear step solver for BlockSparseMatrix systems.
 *
 * Backed by BlockSparseLDLT with AMD ordering on the block quotient graph.
 * Supports Schur complement reduction, ultra-fast independent block-diagonal inversions,
 * and coupled elimination (e.g. distance priors between points).
 */
template <typename BlockMatrix>
struct BlockSparseLinearSolvers {
  using Scalar = typename BlockMatrix::ScalarType;
  using LDLTType = BlockSparseLDLT<BlockMatrix, BlockAMDOrdering>;

  Index dims = 0;
  Index num_reduced_ = 0;
  LinearSolverOptions options{};

  explicit BlockSparseLinearSolvers(const LinearSolverOptions& solver_options = {})
      : options(solver_options) {}

  /// Clears analysis and factorization states.
  void reset() {
    dims = 0;
    num_reduced_ = 0;
    analyzed_ = false;
    computed_ = false;
  }

  /// Resizes the solver dimension and reduced block dimension.
  void resize(Index new_dims, Index num_reduced = 0) {
    if (dims != new_dims || num_reduced_ != num_reduced) {
      dims = new_dims;
      num_reduced_ = num_reduced;
      reset();
    }
  }

  /// Analyzes the sparsity pattern and prepares factorization buffers.
  bool Analyze(const BlockMatrix& H, Index num_reduced = 0) {
    analyzed_ = false;
    computed_ = false;
    const Index dimension = H.rows();
    if (H.cols() != dimension || dimension <= 0 || num_reduced < 0 || num_reduced >= dimension ||
        (num_reduced > 0 && !options.allow_schur) || !IsGroupBoundary(H, num_reduced)) {
      num_reduced_ = -1;
      return false;
    }

    dims = dimension;
    num_reduced_ = num_reduced;

    ldlt_.analyzePattern(H, num_reduced_);
    analyzed_ = ldlt_.isAnalyzed();
    return analyzed_;
  }

  /// Numerical factorization: factors the block-sparse system using BlockSparseLDLT.
  bool Compute(const BlockMatrix& H) {
    computed_ = false;
    if (!analyzed_ || H.rows() != dims || H.cols() != dims || !ldlt_.matchesStructure(H)) {
      return false;
    }
    ldlt_.compute(H);
    computed_ = (ldlt_.info() == Eigen::Success);
    return computed_;
  }

  /// Solves the linear system H * solution = rhs (analyzing and computing if needed).
  std::optional<Vector<Scalar, Dynamic>> Solve(const BlockMatrix& H,
                                               const Vector<Scalar, Dynamic>& rhs) {
    if (num_reduced_ < 0 || num_reduced_ >= H.rows() ||
        (num_reduced_ > 0 && !options.allow_schur)) {
      return std::nullopt;
    }
    if (!analyzed_ || !ldlt_.matchesStructure(H)) {
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

  [[nodiscard]] bool IsAnalyzed() const noexcept { return analyzed_; }
  [[nodiscard]] bool IsComputed() const noexcept { return computed_; }
  [[nodiscard]] LinearSolverMethod GetMethod() const { return options.method; }
  [[nodiscard]] Index Dims() const { return dims; }

 private:
  static bool IsGroupBoundary(const BlockMatrix& H, Index dimension) {
    if (dimension == 0) return true;
    for (size_t type = 0; type < BlockMatrix::kNumBlockTypes; ++type) {
      if (H.base_offset(type) == dimension) return true;
    }
    return false;
  }

  LDLTType ldlt_;
  bool analyzed_ = false;
  bool computed_ = false;
};

}  // namespace tinyopt
