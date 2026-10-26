// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/traits/matrix_traits.h>
#include <tinyopt/types.h>

namespace tinyopt {

/**
 * @brief High-performance symmetric Block Tridiagonal Matrix storing fixed or dynamic size blocks.
 *
 * Stores diagonal blocks D_0, ..., D_{N-1} (size B x B, symmetric) and
 * super-diagonal blocks U_0, ..., U_{N-2} (size B x B, upper off-diagonal).
 * Since the matrix is symmetric, the sub-diagonal blocks are L_i = U_i^T.
 *
 * Contiguous storage in std::vector<Matrix<Scalar, BlockSize, BlockSize>> provides:
 *   - Zero-lookup overhead during accumulation (direct indexing).
 *   - Maximum cache-locality and vectorization.
 *   - In-place LM damping (scaleDiagonal) and Jacobi scaling.
 */
template <typename Scalar_, int BlockSize_ = Dynamic>
class BlockTridiagonalMatrix {
 public:
  using Scalar = Scalar_;
  using ScalarType = Scalar;
  static constexpr int kBlockSize = BlockSize_;

  using Block = Matrix<Scalar, kBlockSize, kBlockSize>;
  using ConstBlockMap = Eigen::Map<const Block>;
  using BlockMap = Eigen::Map<Block>;

  // ---------------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------------

  BlockTridiagonalMatrix() = default;

  explicit BlockTridiagonalMatrix(Index num_blocks,
                                  Index block_size = (kBlockSize != Dynamic ? kBlockSize : 0)) {
    resize(num_blocks, block_size);
  }

  // ---------------------------------------------------------------------------
  // Dimensions & Sizing
  // ---------------------------------------------------------------------------

  void resize(Index num_blocks, Index block_size = (kBlockSize != Dynamic ? kBlockSize : 0)) {
    if (num_blocks < 0) {
      throw std::invalid_argument("Number of blocks must be non-negative");
    }
    if constexpr (kBlockSize != Dynamic) {
      block_size_ = kBlockSize;
    } else {
      if (num_blocks > 0 && block_size <= 0) {
        throw std::invalid_argument("Block size must be positive for dynamic blocks");
      }
      block_size_ = block_size;
    }

    num_blocks_ = num_blocks;
    if (num_blocks_ > 0) {
      diag_blocks_.resize(static_cast<size_t>(num_blocks_));
      for (auto& d : diag_blocks_) {
        if constexpr (kBlockSize == Dynamic) d.resize(block_size_, block_size_);
        d.setZero();
      }
      off_diag_blocks_.resize(static_cast<size_t>(num_blocks_ > 1 ? num_blocks_ - 1 : 0));
      for (auto& u : off_diag_blocks_) {
        if constexpr (kBlockSize == Dynamic) u.resize(block_size_, block_size_);
        u.setZero();
      }
    } else {
      diag_blocks_.clear();
      off_diag_blocks_.clear();
    }
  }

  void resizeFromTotalDims(Index total_dims) {
    if constexpr (kBlockSize != Dynamic) {
      if (total_dims % kBlockSize != 0) {
        throw std::invalid_argument("Total dimensions must be a multiple of block size");
      }
      resize(total_dims / kBlockSize, kBlockSize);
    } else {
      if (block_size_ <= 0) block_size_ = 1;
      resize(total_dims / block_size_, block_size_);
    }
  }

  void clear() noexcept {
    num_blocks_ = 0;
    diag_blocks_.clear();
    off_diag_blocks_.clear();
  }

  void setZero() noexcept {
    for (auto& d : diag_blocks_) d.setZero();
    for (auto& u : off_diag_blocks_) u.setZero();
  }

  void setIdentity() noexcept {
    setZero();
    for (auto& d : diag_blocks_) d.setIdentity();
  }

  [[nodiscard]] Index num_blocks() const noexcept { return num_blocks_; }
  [[nodiscard]] Index num_blocks(size_t) const noexcept { return num_blocks_; }
  [[nodiscard]] Index block_size() const noexcept { return block_size_; }
  [[nodiscard]] Index rows() const noexcept { return num_blocks_ * block_size_; }
  [[nodiscard]] Index cols() const noexcept { return rows(); }
  [[nodiscard]] Index size() const noexcept { return rows() * cols(); }
  [[nodiscard]] Index dims() const noexcept { return rows(); }
  [[nodiscard]] static constexpr size_t num_block_types() noexcept { return 1; }

  [[nodiscard]] bool empty() const noexcept { return num_blocks_ == 0; }

  // ---------------------------------------------------------------------------
  // Block Access (Direct & Fast)
  // ---------------------------------------------------------------------------

  /// Access diagonal block D_i (symmetric B x B)
  [[nodiscard]] Block& diagonal(Index i) { return diag_blocks_[static_cast<size_t>(i)]; }

  [[nodiscard]] const Block& diagonal(Index i) const {
    return diag_blocks_[static_cast<size_t>(i)];
  }

  /// Access super-diagonal block U_i = A(i, i+1) (B x B)
  [[nodiscard]] Block& off_diagonal(Index i) { return off_diag_blocks_[static_cast<size_t>(i)]; }

  [[nodiscard]] const Block& off_diagonal(Index i) const {
    return off_diag_blocks_[static_cast<size_t>(i)];
  }

  /// Block access matching BlockSparseMatrix API: .block<Rows, Cols>(r, c)
  template <int Rows = kBlockSize, int Cols = kBlockSize>
  decltype(auto) block(Index r, Index c) {
    if (r == c) {
      if (r >= num_blocks_) {
        resize(r + 1, block_size_);
      }
      return (diag_blocks_[static_cast<size_t>(r)]);
    } else if (c == r + 1) {
      if (c >= num_blocks_) {
        resize(c + 1, block_size_);
      }
      return (off_diag_blocks_[static_cast<size_t>(r)]);
    } else if (r == c + 1) {
      if (r >= num_blocks_) {
        resize(r + 1, block_size_);
      }
      return off_diag_blocks_[static_cast<size_t>(c)].transpose();
    } else {
      throw std::out_of_range("BlockTridiagonalMatrix only stores blocks with |r - c| <= 1");
    }
  }

  template <int Rows = kBlockSize, int Cols = kBlockSize>
  decltype(auto) block(Index r, Index c) const {
    if (r == c) {
      return (diag_blocks_[static_cast<size_t>(r)]);
    } else if (c == r + 1) {
      return (off_diag_blocks_[static_cast<size_t>(r)]);
    } else if (r == c + 1) {
      return off_diag_blocks_[static_cast<size_t>(c)].transpose();
    } else {
      throw std::out_of_range("BlockTridiagonalMatrix only stores blocks with |r - c| <= 1");
    }
  }

  // ---------------------------------------------------------------------------
  // Vector Storage Access
  // ---------------------------------------------------------------------------

  [[nodiscard]] std::vector<Block>& diag_blocks() noexcept { return diag_blocks_; }
  [[nodiscard]] const std::vector<Block>& diag_blocks() const noexcept { return diag_blocks_; }

  [[nodiscard]] std::vector<Block>& off_diag_blocks() noexcept { return off_diag_blocks_; }
  [[nodiscard]] const std::vector<Block>& off_diag_blocks() const noexcept {
    return off_diag_blocks_;
  }

  // ---------------------------------------------------------------------------
  // Optimization & Solver Methods (LM Damping, Jacobi Scaling, Diagonals)
  // ---------------------------------------------------------------------------

  /// Extract flat diagonal vector of the entire matrix
  [[nodiscard]] Vector<Scalar, Dynamic> diagonal() const {
    Vector<Scalar, Dynamic> diag(rows());
    for (Index i = 0; i < num_blocks_; ++i) {
      diag.segment(i * block_size_, block_size_) = diag_blocks_[static_cast<size_t>(i)].diagonal();
    }
    return diag;
  }

  /// Multiply entire diagonal by a scalar factor: D_i(r, r) *= factor (for Levenberg-Marquardt)
  void scaleDiagonal(Scalar factor) {
    for (auto& d : diag_blocks_) {
      d.diagonal() *= factor;
    }
  }

  /// Apply Jacobi scaling to diagonal and off-diagonal blocks
  void scaleByDiagonal(const Vector<Scalar, Dynamic>& scale, bool inverse = false) {
    for (Index i = 0; i < num_blocks_; ++i) {
      const auto s_i = scale.segment(i * block_size_, block_size_);
      auto& d = diag_blocks_[static_cast<size_t>(i)];
      if (inverse) {
        d.array() /= (s_i * s_i.transpose()).array();
      } else {
        d.array() *= (s_i * s_i.transpose()).array();
      }
    }
    for (Index i = 0; i < num_blocks_ - 1; ++i) {
      const auto s_i = scale.segment(i * block_size_, block_size_);
      const auto s_ip1 = scale.segment((i + 1) * block_size_, block_size_);
      auto& u = off_diag_blocks_[static_cast<size_t>(i)];
      if (inverse) {
        u.array() /= (s_i * s_ip1.transpose()).array();
      } else {
        u.array() *= (s_i * s_ip1.transpose()).array();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Matrix-Vector Product: y = A * x
  // ---------------------------------------------------------------------------

  [[nodiscard]] Vector<Scalar, Dynamic> operator*(const Vector<Scalar, Dynamic>& x) const {
    if (x.size() != rows()) {
      throw std::invalid_argument("BlockTridiagonalMatrix * vector dimension mismatch");
    }
    Vector<Scalar, Dynamic> y = Vector<Scalar, Dynamic>::Zero(rows());
    if (num_blocks_ == 0) return y;

    for (Index i = 0; i < num_blocks_; ++i) {
      const auto x_i = x.segment(i * block_size_, block_size_);
      auto y_i = y.segment(i * block_size_, block_size_);
      y_i.noalias() += diag_blocks_[static_cast<size_t>(i)] * x_i;
      if (i > 0) {
        const auto x_prev = x.segment((i - 1) * block_size_, block_size_);
        y_i.noalias() += off_diag_blocks_[static_cast<size_t>(i - 1)].transpose() * x_prev;
      }
      if (i < num_blocks_ - 1) {
        const auto x_next = x.segment((i + 1) * block_size_, block_size_);
        y_i.noalias() += off_diag_blocks_[static_cast<size_t>(i)] * x_next;
      }
    }
    return y;
  }

  // ---------------------------------------------------------------------------
  // Conversions (Dense & Sparse)
  // ---------------------------------------------------------------------------

  [[nodiscard]] Matrix<Scalar, Dynamic, Dynamic> toDense() const {
    const Index total_dims = rows();
    Matrix<Scalar, Dynamic, Dynamic> mat =
        Matrix<Scalar, Dynamic, Dynamic>::Zero(total_dims, total_dims);
    for (Index i = 0; i < num_blocks_; ++i) {
      const Index off_i = i * block_size_;
      mat.block(off_i, off_i, block_size_, block_size_) = diag_blocks_[static_cast<size_t>(i)];
      if (i < num_blocks_ - 1) {
        const Index off_next = (i + 1) * block_size_;
        mat.block(off_i, off_next, block_size_, block_size_) =
            off_diag_blocks_[static_cast<size_t>(i)];
        mat.block(off_next, off_i, block_size_, block_size_) =
            off_diag_blocks_[static_cast<size_t>(i)].transpose();
      }
    }
    return mat;
  }

  [[nodiscard]] SparseMatrix<Scalar> toSparse() const {
    const Index total_dims = rows();
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(static_cast<size_t>(total_dims * block_size_ * 3));
    for (Index i = 0; i < num_blocks_; ++i) {
      const Index off_i = i * block_size_;
      const auto& d = diag_blocks_[static_cast<size_t>(i)];
      for (int r = 0; r < block_size_; ++r) {
        for (int c = 0; c < block_size_; ++c) {
          if (d(r, c) != Scalar(0)) triplets.emplace_back(off_i + r, off_i + c, d(r, c));
        }
      }
      if (i < num_blocks_ - 1) {
        const Index off_next = (i + 1) * block_size_;
        const auto& u = off_diag_blocks_[static_cast<size_t>(i)];
        for (int r = 0; r < block_size_; ++r) {
          for (int c = 0; c < block_size_; ++c) {
            if (u(r, c) != Scalar(0)) {
              triplets.emplace_back(off_i + r, off_next + c, u(r, c));
              triplets.emplace_back(off_next + c, off_i + r, u(r, c));
            }
          }
        }
      }
    }
    SparseMatrix<Scalar> sp(total_dims, total_dims);
    sp.setFromTriplets(triplets.begin(), triplets.end());
    sp.makeCompressed();
    return sp;
  }

  [[nodiscard]] Scalar squaredNorm() const noexcept {
    Scalar total = 0;
    for (const auto& d : diag_blocks_) total += d.squaredNorm();
    for (const auto& u : off_diag_blocks_) total += Scalar(2) * u.squaredNorm();
    return total;
  }

  [[nodiscard]] Scalar norm() const noexcept {
    using std::sqrt;
    return sqrt(squaredNorm());
  }

  [[nodiscard]] BlockSparseMatrixX<Scalar> toBlockSparseMatX() const {
    return BlockSparseMatrixX<Scalar>(toSparse(), {static_cast<int>(block_size_)}, {num_blocks_});
  }

  [[nodiscard]] const BlockTridiagonalMatrix& transpose() const noexcept { return *this; }

  [[nodiscard]] bool isApprox(const BlockTridiagonalMatrix& other,
                              Scalar prec = Eigen::NumTraits<Scalar>::dummy_precision()) const {
    if (num_blocks_ != other.num_blocks_ || block_size_ != other.block_size_) return false;
    for (size_t i = 0; i < diag_blocks_.size(); ++i) {
      if (!diag_blocks_[i].isApprox(other.diag_blocks_[i], prec)) return false;
    }
    for (size_t i = 0; i < off_diag_blocks_.size(); ++i) {
      if (!off_diag_blocks_[i].isApprox(other.off_diag_blocks_[i], prec)) return false;
    }
    return true;
  }

 private:
  Index num_blocks_ = 0;
  Index block_size_ = (kBlockSize != Dynamic ? kBlockSize : 0);
  std::vector<Block> diag_blocks_;
  std::vector<Block> off_diag_blocks_;
};

// Aliases
template <int BlockSize = Dynamic>
using BlockTridiagonalMat = BlockTridiagonalMatrix<double, BlockSize>;

template <int BlockSize = Dynamic>
using BlockTridiagonalMatf = BlockTridiagonalMatrix<float, BlockSize>;

template <typename Scalar = double>
using BlockTridiagonalMatrixX = BlockTridiagonalMatrix<Scalar, Dynamic>;

namespace traits {
template <typename Scalar, int BlockSize>
struct is_block_tridiagonal_matrix<BlockTridiagonalMatrix<Scalar, BlockSize>> : std::true_type {};
}  // namespace traits

}  // namespace tinyopt
