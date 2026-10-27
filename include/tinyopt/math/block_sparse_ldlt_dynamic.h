// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>

#include <Eigen/Core>

#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/types.h>

namespace tinyopt {

/**
 * @brief Runtime block-sparse LDLT for `BlockSparseMatrix<Scalar, Dynamic>`.
 *
 * Block sizes and counts are known only at analyze time. Symbolic analysis builds the block
 * elimination pattern; the numerical phase uses preallocated dense block buffers and per-block
 * LLT inverses, so repeated compute/solve calls avoid heap allocation. The matrix is expected to
 * store the symmetric upper triangle (lower-triangular writes are mirrored by setBlock()).
 *
 * Schur reduction and guided orderings are not supported yet; `num_reduced` must be zero.
 */
template <typename BlockMatrix>
class RuntimeBlockSparseLDLT {
 public:
  using Scalar = typename BlockMatrix::ScalarType;
  using SparseMatrixType = typename BlockMatrix::SparseMatrixType;
  using VectorType = Vector<Scalar, Dynamic>;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using StridedMap = Eigen::Map<MatrixType, 0, Eigen::OuterStride<>>;

  RuntimeBlockSparseLDLT() = default;

  RuntimeBlockSparseLDLT& analyzePattern(const BlockMatrix& H, Index num_reduced = 0) {
    analyzed_ = false;
    computed_ = false;
    info_ = Eigen::Success;
    if (num_reduced != 0) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    block_sizes_ = H.block_sizes();
    block_counts_ = H.block_counts();
    if (block_sizes_.empty() || block_sizes_.size() != block_counts_.size()) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    num_blocks_ = static_cast<int>(H.num_blocks());
    dims_ = H.rows();
    block_size_of_node_.resize(static_cast<size_t>(num_blocks_));
    scalar_offset_.assign(static_cast<size_t>(num_blocks_) + 1, 0);
    block_of_scalar_.resize(static_cast<size_t>(dims_));

    Index offset = 0;
    int node = 0;
    max_block_size_ = 1;
    for (size_t type = 0; type < block_sizes_.size(); ++type) {
      const int block_size = block_sizes_[type];
      max_block_size_ = std::max(max_block_size_, block_size);
      for (Index local = 0; local < block_counts_[type]; ++local, ++node) {
        block_size_of_node_[static_cast<size_t>(node)] = block_size;
        scalar_offset_[static_cast<size_t>(node)] = offset;
        for (int i = 0; i < block_size; ++i)
          block_of_scalar_[static_cast<size_t>(offset + i)] = node;
        offset += block_size;
      }
    }
    scalar_offset_[static_cast<size_t>(num_blocks_)] = offset;
    if (offset != dims_) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    upper_pairs_.clear();
    a_blocks_.clear();
    l_blocks_.clear();
    const auto add_upper_pair = [&](int i, int j) {
      if (i > j) std::swap(i, j);
      const uint64_t key = Key(i, j);
      if (a_blocks_.find(key) != a_blocks_.end()) return;
      const int rows = block_size_of_node_[static_cast<size_t>(i)];
      const int cols = block_size_of_node_[static_cast<size_t>(j)];
      a_blocks_.emplace(key, MatrixType::Zero(rows, cols));
      if (i != j) a_blocks_.emplace(Key(j, i), MatrixType::Zero(cols, rows));
      upper_pairs_.emplace_back(i, j);
    };

    const SparseMatrixType& sparse = H.toSparse();
    for (int outer = 0; outer < sparse.outerSize(); ++outer) {
      for (typename SparseMatrixType::InnerIterator it(sparse, outer); it; ++it) {
        add_upper_pair(block_of_scalar_[static_cast<size_t>(it.row())],
                       block_of_scalar_[static_cast<size_t>(it.col())]);
      }
    }
    for (int i = 0; i < num_blocks_; ++i) add_upper_pair(i, i);
    std::sort(upper_pairs_.begin(), upper_pairs_.end());

    col_rows_.assign(static_cast<size_t>(num_blocks_), {});
    for (const auto& pair : upper_pairs_) {
      if (pair.first != pair.second)
        col_rows_[static_cast<size_t>(pair.first)].push_back(pair.second);
    }
    for (int column = 0; column < num_blocks_; ++column) {
      std::set<int> rows(col_rows_[static_cast<size_t>(column)].begin(),
                         col_rows_[static_cast<size_t>(column)].end());
      bool changed = true;
      while (changed) {
        changed = false;
        const std::vector<int> current(rows.begin(), rows.end());
        for (const int row : current) {
          if (row >= column) continue;
          for (const int fill : col_rows_[static_cast<size_t>(row)]) {
            if (fill > column && rows.insert(fill).second) changed = true;
          }
        }
      }
      col_rows_[static_cast<size_t>(column)].assign(rows.begin(), rows.end());
    }

    lower_pattern_.assign(static_cast<size_t>(num_blocks_), {});
    for (int column = 0; column < num_blocks_; ++column) {
      for (const int row : col_rows_[static_cast<size_t>(column)]) {
        lower_pattern_[static_cast<size_t>(row)].push_back(column);
        const int rows = block_size_of_node_[static_cast<size_t>(row)];
        const int cols = block_size_of_node_[static_cast<size_t>(column)];
        l_blocks_.emplace(Key(row, column), MatrixType::Zero(rows, cols));
      }
    }
    for (auto& pattern : lower_pattern_) std::sort(pattern.begin(), pattern.end());

    d_.resize(static_cast<size_t>(num_blocks_));
    d_inverse_.resize(static_cast<size_t>(num_blocks_));
    llt_.resize(static_cast<size_t>(num_blocks_));
    for (int i = 0; i < num_blocks_; ++i) {
      const int size = block_size_of_node_[static_cast<size_t>(i)];
      d_[static_cast<size_t>(i)] = MatrixType::Zero(size, size);
      d_inverse_[static_cast<size_t>(i)] = MatrixType::Zero(size, size);
      // Warm up the factorization buffers so the numerical phase does not allocate.
      llt_[static_cast<size_t>(i)].compute(MatrixType::Identity(size, size));
    }
    scratch1_ = MatrixType::Zero(max_block_size_, max_block_size_);
    scratch2_ = MatrixType::Zero(max_block_size_, max_block_size_);
    scratch3_ = MatrixType::Zero(max_block_size_, max_block_size_);

    analyzed_ = true;
    return *this;
  }

  RuntimeBlockSparseLDLT& compute(const BlockMatrix& H) {
    computed_ = false;
    info_ = Eigen::Success;
    if (!analyzed_ || H.rows() != dims_ || H.cols() != dims_ || !matchesStructure(H)) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    for (auto& entry : a_blocks_) entry.second.setZero();
    const SparseMatrixType& sparse = H.toSparse();
    for (int outer = 0; outer < sparse.outerSize(); ++outer) {
      for (typename SparseMatrixType::InnerIterator it(sparse, outer); it; ++it) {
        // The matrix is symmetric; read one triangle and reconstruct the other.
        if (it.row() > it.col()) continue;
        const int row_node = block_of_scalar_[static_cast<size_t>(it.row())];
        const int col_node = block_of_scalar_[static_cast<size_t>(it.col())];
        const Index row_local = it.row() - scalar_offset_[static_cast<size_t>(row_node)];
        const Index col_local = it.col() - scalar_offset_[static_cast<size_t>(col_node)];
        a_blocks_[Key(row_node, col_node)](row_local, col_local) += it.value();
        if (row_node != col_node) {
          a_blocks_[Key(col_node, row_node)](col_local, row_local) += it.value();
        } else if (row_local != col_local) {
          a_blocks_[Key(row_node, col_node)](col_local, row_local) += it.value();
        }
      }
    }

    for (int column = 0; column < num_blocks_; ++column) {
      const int size = block_size_of_node_[static_cast<size_t>(column)];
      MatrixType& diagonal = d_[static_cast<size_t>(column)];
      diagonal = a_blocks_[Key(column, column)];
      for (const int k : lower_pattern_[static_cast<size_t>(column)]) {
        const int k_size = block_size_of_node_[static_cast<size_t>(k)];
        const MatrixType& l_jk = l_blocks_[Key(column, k)];
        StridedMap tmp1(scratch1_.data(), size, k_size, Eigen::OuterStride<>(max_block_size_));
        StridedMap tmp2(scratch2_.data(), size, size, Eigen::OuterStride<>(max_block_size_));
        tmp1.noalias() = l_jk * d_[static_cast<size_t>(k)];
        tmp2.noalias() = tmp1 * l_jk.transpose();
        diagonal.noalias() -= tmp2;
      }
      if (!diagonal.allFinite()) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }
      MatrixType& inverse = d_inverse_[static_cast<size_t>(column)];
      inverse.setIdentity(size, size);
      llt_[static_cast<size_t>(column)].compute(diagonal);
      if (llt_[static_cast<size_t>(column)].info() != Eigen::Success) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }
      llt_[static_cast<size_t>(column)].solveInPlace(inverse);
      for (const int row : col_rows_[static_cast<size_t>(column)]) {
        const int row_size = block_size_of_node_[static_cast<size_t>(row)];
        MatrixType& l_ij = l_blocks_[Key(row, column)];
        StridedMap sum(scratch3_.data(), row_size, size, Eigen::OuterStride<>(max_block_size_));
        sum = a_blocks_[Key(row, column)];
        for (const int k : lower_pattern_[static_cast<size_t>(column)]) {
          const auto l_ik = l_blocks_.find(Key(row, k));
          if (l_ik == l_blocks_.end()) continue;
          const int k_size = block_size_of_node_[static_cast<size_t>(k)];
          const MatrixType& l_jk = l_blocks_[Key(column, k)];
          StridedMap tmp1(scratch1_.data(), row_size, k_size,
                          Eigen::OuterStride<>(max_block_size_));
          StridedMap tmp2(scratch2_.data(), row_size, size, Eigen::OuterStride<>(max_block_size_));
          tmp1.noalias() = l_ik->second * d_[static_cast<size_t>(k)];
          tmp2.noalias() = tmp1 * l_jk.transpose();
          sum.noalias() -= tmp2;
        }
        l_ij.noalias() = sum * d_inverse_[static_cast<size_t>(column)];
        if (!l_ij.allFinite()) {
          info_ = Eigen::NumericalIssue;
          return *this;
        }
      }
    }

    computed_ = true;
    return *this;
  }

  [[nodiscard]] bool matchesStructure(const BlockMatrix& H) const {
    if (H.rows() != dims_ || H.cols() != dims_ || H.block_sizes() != block_sizes_ ||
        H.block_counts() != block_counts_)
      return false;
    std::vector<std::pair<int, int>> pairs;
    const SparseMatrixType& sparse = H.toSparse();
    for (int outer = 0; outer < sparse.outerSize(); ++outer) {
      for (typename SparseMatrixType::InnerIterator it(sparse, outer); it; ++it) {
        int row = block_of_scalar_[static_cast<size_t>(it.row())];
        int col = block_of_scalar_[static_cast<size_t>(it.col())];
        if (row > col) std::swap(row, col);
        pairs.emplace_back(row, col);
      }
    }
    for (int i = 0; i < num_blocks_; ++i) pairs.emplace_back(i, i);
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    return pairs == upper_pairs_;
  }

  [[nodiscard]] bool isAnalyzed() const noexcept { return analyzed_; }
  [[nodiscard]] bool isComputed() const noexcept { return computed_; }
  [[nodiscard]] Eigen::ComputationInfo info() const noexcept { return info_; }
  [[nodiscard]] Index dims() const noexcept { return dims_; }

  [[nodiscard]] VectorType solve(const VectorType& rhs) const {
    VectorType result = rhs;
    solveInPlace(result);
    return result;
  }

  [[nodiscard]] MatrixType solve(const MatrixType& rhs) const {
    MatrixType result = rhs;
    VectorType column(rhs.rows());
    for (Index c = 0; c < rhs.cols(); ++c) {
      column = rhs.col(c);
      solveInPlace(column);
      result.col(c) = column;
    }
    return result;
  }

 private:
  static uint64_t Key(int row, int col) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) | static_cast<uint32_t>(col);
  }

  void solveInPlace(VectorType& rhs) const {
    // Forward substitution: L z = b.
    for (int column = 0; column < num_blocks_; ++column) {
      const int size = block_size_of_node_[static_cast<size_t>(column)];
      const Index offset = scalar_offset_[static_cast<size_t>(column)];
      for (const int row : col_rows_[static_cast<size_t>(column)]) {
        const int row_size = block_size_of_node_[static_cast<size_t>(row)];
        const Index row_offset = scalar_offset_[static_cast<size_t>(row)];
        rhs.segment(row_offset, row_size).noalias() -=
            l_blocks_.at(Key(row, column)) * rhs.segment(offset, size);
      }
    }
    // Diagonal solve: y = D^-1 z.
    for (int column = 0; column < num_blocks_; ++column) {
      const int size = block_size_of_node_[static_cast<size_t>(column)];
      const Index offset = scalar_offset_[static_cast<size_t>(column)];
      rhs.segment(offset, size) =
          d_inverse_[static_cast<size_t>(column)] * rhs.segment(offset, size);
    }
    // Back substitution: L^T x = y.
    for (int column = num_blocks_ - 1; column >= 0; --column) {
      const int size = block_size_of_node_[static_cast<size_t>(column)];
      const Index offset = scalar_offset_[static_cast<size_t>(column)];
      for (const int row : col_rows_[static_cast<size_t>(column)]) {
        const int row_size = block_size_of_node_[static_cast<size_t>(row)];
        const Index row_offset = scalar_offset_[static_cast<size_t>(row)];
        rhs.segment(offset, size).noalias() -=
            l_blocks_.at(Key(row, column)).transpose() * rhs.segment(row_offset, row_size);
      }
    }
  }

  std::vector<int> block_sizes_;
  std::vector<Index> block_counts_;
  std::vector<int> block_size_of_node_;
  std::vector<Index> scalar_offset_;
  std::vector<int> block_of_scalar_;
  std::vector<std::pair<int, int>> upper_pairs_;
  std::vector<std::vector<int>> col_rows_;
  std::vector<std::vector<int>> lower_pattern_;
  std::unordered_map<uint64_t, MatrixType> a_blocks_;
  std::unordered_map<uint64_t, MatrixType> l_blocks_;
  std::vector<MatrixType> d_;
  std::vector<MatrixType> d_inverse_;
  std::vector<Eigen::LLT<MatrixType>> llt_;
  MatrixType scratch1_;
  MatrixType scratch2_;
  MatrixType scratch3_;
  int num_blocks_ = 0;
  int max_block_size_ = 1;
  Index dims_ = 0;
  Eigen::ComputationInfo info_ = Eigen::Success;
  bool analyzed_ = false;
  bool computed_ = false;
};

}  // namespace tinyopt
