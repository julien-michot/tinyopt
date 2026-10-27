// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <tinyopt/math/block_tridiagonal_matrix.h>
#include <tinyopt/types.h>

namespace tinyopt {

/**
 * @brief High-performance Block Tridiagonal LDLT / Cholesky Decomposition.
 *
 * Factors a symmetric positive-definite block tridiagonal matrix A into A = L * Delta * L^T:
 *   - Delta = diag(Delta_0, ..., Delta_{N-1}) (block diagonal)
 *   - L is block unit lower bidiagonal with subdiagonal blocks Gamma_i = U_i^T * Delta_i^{-1}
 *   - L^T has superdiagonal blocks C_i = Delta_i^{-1} * U_i
 *
 * Solves Ax = b in O(N) operations with 0 dynamic allocations during compute() and solve().
 * Supports both BlockTridiagonalMatrix and raw block pointer sequences (zero-copy from
 * BlockSparseMatrix).
 */
template <typename Scalar_, int BlockSize_ = Dynamic>
class BlockTridiagonalLDLT {
 public:
  using Scalar = Scalar_;
  static constexpr int kBlockSize = BlockSize_;

  using Block = Matrix<Scalar, kBlockSize, kBlockSize>;
  using VectorType = Vector<Scalar, Dynamic>;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;

  BlockTridiagonalLDLT() = default;

  explicit BlockTridiagonalLDLT(Index num_blocks,
                                Index block_size = (kBlockSize != Dynamic ? kBlockSize : 0)) {
    analyzePattern(num_blocks, block_size);
  }

  template <int MatBlockSize>
  explicit BlockTridiagonalLDLT(const BlockTridiagonalMatrix<Scalar, MatBlockSize>& mat) {
    analyzePattern(mat.num_blocks(), mat.block_size());
    if (analyzed_) compute(mat);
  }

  // ---------------------------------------------------------------------------
  // Symbolic Phase
  // ---------------------------------------------------------------------------

  BlockTridiagonalLDLT& analyzePattern(Index num_blocks,
                                       Index block_size = (kBlockSize != Dynamic ? kBlockSize
                                                                                 : 0)) {
    analyzed_ = false;
    computed_ = false;
    info_ = Eigen::Success;

    if (num_blocks < 0) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    if constexpr (kBlockSize != Dynamic) {
      block_size_ = kBlockSize;
    } else {
      block_size_ = block_size;
    }

    num_blocks_ = num_blocks;
    dims_ = num_blocks_ * block_size_;

    if (num_blocks_ > 0) {
      inv_delta_.resize(static_cast<size_t>(num_blocks_));
      for (auto& blk : inv_delta_) {
        if constexpr (kBlockSize == Dynamic) blk.resize(block_size_, block_size_);
      }
      C_.resize(static_cast<size_t>(num_blocks_ > 1 ? num_blocks_ - 1 : 0));
      for (auto& blk : C_) {
        if constexpr (kBlockSize == Dynamic) blk.resize(block_size_, block_size_);
      }
      scratch_z_.resize(dims_);
    } else {
      inv_delta_.clear();
      C_.clear();
      scratch_z_.resize(0);
    }

    analyzed_ = true;
    return *this;
  }

  template <int MatBlockSize>
  BlockTridiagonalLDLT& analyzePattern(const BlockTridiagonalMatrix<Scalar, MatBlockSize>& mat) {
    return analyzePattern(mat.num_blocks(), mat.block_size());
  }

  // ---------------------------------------------------------------------------
  // Numerical Factorization: compute()
  // ---------------------------------------------------------------------------

  /// Factors a BlockTridiagonalMatrix directly
  template <int MatBlockSize>
  BlockTridiagonalLDLT& compute(const BlockTridiagonalMatrix<Scalar, MatBlockSize>& mat) {
    if (!analyzed_ || mat.num_blocks() != num_blocks_ || mat.block_size() != block_size_) {
      analyzePattern(mat.num_blocks(), mat.block_size());
    }
    return computeFromBlocks(mat.diag_blocks(), mat.off_diag_blocks());
  }

  /// Factors raw block pointers (used by BlockSparseLDLT for zero-copy elimination)
  template <typename BlockType>
  BlockTridiagonalLDLT& computeFromPointers(const std::vector<const BlockType*>& diag_ptrs,
                                            const std::vector<const BlockType*>& off_diag_ptrs) {
    computed_ = false;
    info_ = Eigen::Success;

    if (!analyzed_ || static_cast<Index>(diag_ptrs.size()) != num_blocks_) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    if (num_blocks_ == 0) {
      computed_ = true;
      return *this;
    }

    // Step 0: Invert Delta_0 = D_0
    Block delta = *diag_ptrs[0];
    if (!InvertAndStore(delta, inv_delta_[0])) {
      info_ = Eigen::NumericalIssue;
      return *this;
    }

    // Step 1 to N-1: Recurrence
    for (Index i = 0; i < num_blocks_ - 1; ++i) {
      const auto& U_i = *off_diag_ptrs[static_cast<size_t>(i)];
      if (!U_i.allFinite()) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }

      // C_i = Delta_i^{-1} * U_i
      Block& C_i = C_[static_cast<size_t>(i)];
      C_i.noalias() = inv_delta_[static_cast<size_t>(i)] * U_i;

      // Delta_{i+1} = D_{i+1} - U_i^T * C_i
      delta = *diag_ptrs[static_cast<size_t>(i + 1)];
      delta.noalias() -= U_i.transpose() * C_i;

      // Invert Delta_{i+1}
      if (!InvertAndStore(delta, inv_delta_[static_cast<size_t>(i + 1)])) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }
    }

    computed_ = true;
    return *this;
  }

  // ---------------------------------------------------------------------------
  // Linear Solve: solve()
  // ---------------------------------------------------------------------------

  /// Solves Ax = rhs for a vector right-hand side
  [[nodiscard]] VectorType solve(const VectorType& rhs) const {
    VectorType solution(dims_);
    if (!computed_ || rhs.size() != dims_ || !rhs.allFinite()) {
      solution.setConstant(std::numeric_limits<Scalar>::quiet_NaN());
      return solution;
    }
    if (num_blocks_ == 0) return solution;

    solveVectorInternal(rhs, solution);
    return solution;
  }

  /// Solves AX = B for a matrix right-hand side (e.g. B = I for inverse recovery)
  [[nodiscard]] MatrixType solve(const MatrixType& B) const {
    MatrixType X(dims_, B.cols());
    if (!computed_ || B.rows() != dims_ || !B.allFinite()) {
      X.setConstant(std::numeric_limits<Scalar>::quiet_NaN());
      return X;
    }
    if (num_blocks_ == 0) return X;

    solveMatrixInternal(B, X);
    return X;
  }

  /// Eigen-compatible generic MatrixBase solve overload
  template <typename Derived>
  auto solve(const Eigen::MatrixBase<Derived>& b) const {
    if constexpr (Derived::ColsAtCompileTime == 1) {
      return solve(VectorType(b));
    } else {
      return solve(MatrixType(b));
    }
  }

  /// Tinyopt-style optional solver returning std::nullopt on failure
  template <typename Rhs>
  auto Solve(const Rhs& b) const -> std::optional<decltype(solve(b))> {
    if (!computed_ || info_ != Eigen::Success) return std::nullopt;
    auto sol = solve(b);
    if (!sol.allFinite()) return std::nullopt;
    return sol;
  }

  /// Computes the block tridiagonal covariance (marginal diagonal blocks and step transitions)
  /// in exact O(N) operations via Takahashi's backward recurrence. Zero dynamic allocations for
  /// fixed-size.
  template <int OutBlockSize = kBlockSize>
  [[nodiscard]] std::optional<BlockTridiagonalMatrix<Scalar, OutBlockSize>>
  inverseBlockTridiagonal() const {
    if (!computed_ || info_ != Eigen::Success) return std::nullopt;
    const int B = (kBlockSize != Dynamic) ? kBlockSize : static_cast<int>(block_size_);
    BlockTridiagonalMatrix<Scalar, OutBlockSize> cov(num_blocks_, B);
    if (num_blocks_ == 0) return cov;

    // Terminal condition: Sigma_{N-1, N-1} = Delta_{N-1}^-1
    cov.diagonal(num_blocks_ - 1) = inv_delta_[static_cast<size_t>(num_blocks_ - 1)];

    // Backward recurrence:
    // Sigma_{i, i+1} = -C_i * Sigma_{i+1, i+1}
    // Sigma_{i, i} = Delta_i^-1 - Sigma_{i, i+1} * C_i^T
    for (Index i = num_blocks_ - 2; i >= 0; --i) {
      const auto& C_i = C_[static_cast<size_t>(i)];
      const auto& next_diag = cov.diagonal(i + 1);
      cov.off_diagonal(i).noalias() = -C_i * next_diag;
      cov.diagonal(i) = inv_delta_[static_cast<size_t>(i)];
      cov.diagonal(i).noalias() -= cov.off_diagonal(i) * C_i.transpose();
    }
    return cov;
  }

  /// Computes the exact covariance block (i, j) in O(|i - j|) operations without full inversion.
  [[nodiscard]] std::optional<Block> blockCovariance(Index i, Index j) const {
    if (!computed_ || info_ != Eigen::Success || i < 0 || i >= num_blocks_ || j < 0 ||
        j >= num_blocks_) {
      return std::nullopt;
    }
    auto maybe_cov = inverseBlockTridiagonal();
    if (!maybe_cov) return std::nullopt;
    if (i == j) {
      return maybe_cov->diagonal(i);
    }
    if (i > j) {
      auto maybe_blk = blockCovariance(j, i);
      if (!maybe_blk) return std::nullopt;
      return maybe_blk->transpose();
    }
    // i < j: Sigma_{i, j} = (-C_i) * (-C_{i+1}) * ... * (-C_{j-1}) * Sigma_{j, j}
    Block result = maybe_cov->diagonal(j);
    for (Index k = j - 1; k >= i; --k) {
      result = (-C_[static_cast<size_t>(k)] * result).eval();
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Queries
  // ---------------------------------------------------------------------------

  [[nodiscard]] Eigen::ComputationInfo info() const noexcept { return info_; }
  [[nodiscard]] bool isAnalyzed() const noexcept { return analyzed_; }
  [[nodiscard]] bool isComputed() const noexcept { return computed_; }
  [[nodiscard]] Index rows() const noexcept { return dims_; }
  [[nodiscard]] Index cols() const noexcept { return dims_; }
  [[nodiscard]] Index num_blocks() const noexcept { return num_blocks_; }
  [[nodiscard]] Index block_size() const noexcept { return block_size_; }

  [[nodiscard]] const std::vector<Block>& inverted_deltas() const noexcept { return inv_delta_; }
  [[nodiscard]] const std::vector<Block>& factor_C() const noexcept { return C_; }

 private:
  template <typename BlockContainer>
  BlockTridiagonalLDLT& computeFromBlocks(const BlockContainer& diag_blocks,
                                          const BlockContainer& off_diag_blocks) {
    computed_ = false;
    info_ = Eigen::Success;

    if (!analyzed_ || static_cast<Index>(diag_blocks.size()) != num_blocks_) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    if (num_blocks_ == 0) {
      computed_ = true;
      return *this;
    }

    // Step 0: Invert Delta_0 = D_0
    Block delta = diag_blocks[0];
    if (!InvertAndStore(delta, inv_delta_[0])) {
      info_ = Eigen::NumericalIssue;
      return *this;
    }

    // Step 1 to N-1: Recurrence
    for (Index i = 0; i < num_blocks_ - 1; ++i) {
      const Block& U_i = off_diag_blocks[static_cast<size_t>(i)];
      if (!U_i.allFinite()) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }

      // C_i = Delta_i^{-1} * U_i
      Block& C_i = C_[static_cast<size_t>(i)];
      C_i.noalias() = inv_delta_[static_cast<size_t>(i)] * U_i;

      // Delta_{i+1} = D_{i+1} - U_i^T * C_i
      delta = diag_blocks[static_cast<size_t>(i + 1)];
      delta.noalias() -= U_i.transpose() * C_i;

      // Invert Delta_{i+1}
      if (!InvertAndStore(delta, inv_delta_[static_cast<size_t>(i + 1)])) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }
    }

    computed_ = true;
    return *this;
  }

  template <typename InputBlock>
  static bool InvertAndStore(const InputBlock& block, Block& inverse) {
    if (!block.allFinite()) return false;
    if (block.rows() <= 4) {
      inverse = block.inverse();
      if (inverse.allFinite()) return true;
    }

    const auto factor = block.template selfadjointView<Eigen::Upper>().ldlt();
    if (factor.info() != Eigen::Success) return false;
    const auto diagonal = factor.vectorD();
    if (!diagonal.allFinite() ||
        diagonal.minCoeff() <= Eigen::NumTraits<Scalar>::epsilon() * diagonal.maxCoeff() ||
        diagonal.minCoeff() <= Scalar(0)) {
      return false;
    }
    inverse = factor.solve(Block::Identity(block.rows(), block.cols()));
    return inverse.allFinite();
  }

  void solveVectorInternal(const VectorType& rhs, VectorType& solution) const {
    const Index bs = block_size_;

    // Phase 1: Forward sweep L * Y = rhs
    scratch_z_.segment(0, bs) = rhs.segment(0, bs);
    for (Index i = 1; i < num_blocks_; ++i) {
      const Index off_curr = i * bs;
      const Index off_prev = (i - 1) * bs;
      scratch_z_.segment(off_curr, bs) = rhs.segment(off_curr, bs);
      scratch_z_.segment(off_curr, bs).noalias() -=
          C_[static_cast<size_t>(i - 1)].transpose() * scratch_z_.segment(off_prev, bs);
    }

    // Phase 2: Diagonal scaling Z = Delta^{-1} * Y
    for (Index i = 0; i < num_blocks_; ++i) {
      const Index off = i * bs;
      solution.segment(off, bs).noalias() =
          inv_delta_[static_cast<size_t>(i)] * scratch_z_.segment(off, bs);
    }

    // Phase 3: Backward sweep L^T * x = Z
    for (Index i = num_blocks_ - 2; i >= 0; --i) {
      const Index off_curr = i * bs;
      const Index off_next = (i + 1) * bs;
      solution.segment(off_curr, bs).noalias() -=
          C_[static_cast<size_t>(i)] * solution.segment(off_next, bs);
    }
  }

  void solveMatrixInternal(const MatrixType& B, MatrixType& X) const {
    const Index bs = block_size_;
    const Index cols = B.cols();
    if (scratch_Z_.rows() != dims_ || scratch_Z_.cols() != cols) {
      scratch_Z_.resize(dims_, cols);
    }

    // Phase 1: Forward sweep L * Y = B
    scratch_Z_.middleRows(0, bs) = B.middleRows(0, bs);
    for (Index i = 1; i < num_blocks_; ++i) {
      const Index off_curr = i * bs;
      const Index off_prev = (i - 1) * bs;
      scratch_Z_.middleRows(off_curr, bs) = B.middleRows(off_curr, bs);
      scratch_Z_.middleRows(off_curr, bs).noalias() -=
          C_[static_cast<size_t>(i - 1)].transpose() * scratch_Z_.middleRows(off_prev, bs);
    }

    // Phase 2: Diagonal scaling Z = Delta^{-1} * Y
    for (Index i = 0; i < num_blocks_; ++i) {
      const Index off = i * bs;
      X.middleRows(off, bs).noalias() =
          inv_delta_[static_cast<size_t>(i)] * scratch_Z_.middleRows(off, bs);
    }

    // Phase 3: Backward sweep L^T * X = Z
    for (Index i = num_blocks_ - 2; i >= 0; --i) {
      const Index off_curr = i * bs;
      const Index off_next = (i + 1) * bs;
      X.middleRows(off_curr, bs).noalias() -=
          C_[static_cast<size_t>(i)] * X.middleRows(off_next, bs);
    }
  }

  Index num_blocks_ = 0;
  Index block_size_ = (kBlockSize != Dynamic ? kBlockSize : 0);
  Index dims_ = 0;
  bool analyzed_ = false;
  bool computed_ = false;
  Eigen::ComputationInfo info_ = Eigen::Success;

  std::vector<Block> inv_delta_;
  std::vector<Block> C_;
  mutable VectorType scratch_z_;
  mutable MatrixType scratch_Z_;
};

}  // namespace tinyopt
