// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>

#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/types.h>

namespace tinyopt {

/// Ordering tag requesting Automatic Minimum Degree (AMD) ordering on the block quotient graph.
struct BlockAMDOrdering {};

/// Ordering tag requesting natural (identity) block ordering.
struct BlockNaturalOrdering {};

/// Ordering tag guiding specific block types to be eliminated first (e.g. points in BA).
template <size_t... TypeOrder>
struct BlockGuidedOrdering {
  static constexpr size_t kNumGuided = sizeof...(TypeOrder);
  static constexpr std::array<size_t, kNumGuided> kOrder = {TypeOrder...};
};

namespace internal {

template <typename T>
struct is_block_guided : std::false_type {};

template <size_t... TypeOrder>
struct is_block_guided<BlockGuidedOrdering<TypeOrder...>> : std::true_type {};

template <typename T>
inline constexpr bool is_block_guided_v = is_block_guided<T>::value;

}  // namespace internal

/**
 * @brief High-performance Block-Sparse LDLT / Cholesky decomposition.
 *
 * Provides an Eigen-compatible decomposition interface (analyzePattern, compute, solve, info)
 * specialized for BlockSparseMatrix<Scalar, BlockSizes...>.
 *
 * Key capabilities:
 *   - Supports AMD ordering, natural ordering, and compile-time guided ordering (e.g.
 * BlockGuidedOrdering<1> for points in BA).
 *   - Automatically partitions eliminated variables into independent blocks (I) and coupled blocks
 * (C):
 *       - Independent blocks (I) undergo ultra-fast, closed-form fixed-size inversion and direct
 * Schur accumulation.
 *       - Coupled blocks (C, e.g. points connected by distance priors) are isolated into a small
 * sparse LDLT.
 *   - Zero dynamic memory allocations during repeated compute() calls.
 *   - solve() accepts both Vector (Ax = b) and Matrix (AX = B) right-hand sides to recover full (B
 * = I) or partial inverses.
 */
template <typename BlockMatrix, typename Ordering = BlockAMDOrdering>
class BlockSparseLDLT {
 public:
  using Scalar = typename BlockMatrix::ScalarType;
  using SparseMatrixType = SparseMatrix<Scalar>;
  using DenseMatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVectorType = Vector<Scalar, Dynamic>;
  using OrderingType = Ordering;

  static constexpr size_t kNumBlockTypes = BlockMatrix::kNumBlockTypes;
  static constexpr size_t kNumBlockPairs = BlockMatrix::kNumPairs;

  /// Represents a connection between an eliminated block and a reduced row block.
  struct CouplingBlock {
    Index row_block = 0;
    size_t matrix_block = 0;
  };

  /// Index mapping into a flat SparseMatrix value buffer.
  struct MatrixValueSlot {
    Index first = -1;
    Index second = -1;
  };

  template <size_t... Is>
  static auto MakeInversesTuple(std::index_sequence<Is...>)
      -> std::tuple<std::vector<
          Matrix<Scalar, BlockMatrix::kBlockSizes[Is], BlockMatrix::kBlockSizes[Is]>>...>;

  using InversesTupleType = decltype(MakeInversesTuple(std::make_index_sequence<kNumBlockTypes>{}));

  BlockSparseLDLT() = default;

  explicit BlockSparseLDLT(const BlockMatrix& H, Index num_reduced = -1) {
    analyzePattern(H, num_reduced);
    if (analyzed_) compute(H);
  }

  /// Symbolic analysis: inspects sparsity, partitions eliminated blocks, and preallocates buffers.
  BlockSparseLDLT& analyzePattern(const BlockMatrix& H, Index num_reduced = -1) {
    analyzed_ = false;
    computed_ = false;
    info_ = Eigen::Success;
    dims_ = H.rows();

    if (H.cols() != dims_ || dims_ <= 0) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    // Determine the partitioning boundary between reduced (cameras) and eliminated (points)
    Index determined_reduced = num_reduced;
    if (determined_reduced < 0) {
      if constexpr (internal::is_block_guided_v<Ordering>) {
        constexpr size_t guided_type = Ordering::kOrder[0];
        if constexpr (guided_type < kNumBlockTypes) {
          determined_reduced = H.base_offset(guided_type);
        } else {
          determined_reduced = 0;
        }
      } else {
        if constexpr (kNumBlockTypes == 2) {
          determined_reduced = H.base_offset(1);
        } else {
          determined_reduced = 0;
        }
      }
    }

    if (determined_reduced < 0 || determined_reduced > dims_ ||
        (determined_reduced > 0 && !IsGroupBoundary(H, determined_reduced))) {
      determined_reduced = 0;
    }

    num_reduced_ = determined_reduced;
    num_eliminated_ = dims_ - num_reduced_;

    // Reset internal structures
    for (auto& s : structure_) s.clear();
    for (auto& b : coupling_blocks_) b.clear();
    for (auto& s : full_value_slots_) s.clear();
    for (auto& s : elim_value_slots_) s.clear();
    std::apply([](auto&... vecs) { (vecs.clear(), ...); }, eliminated_inverses_);
    for (size_t t = 0; t < kNumBlockTypes; ++t) {
      is_block_coupled_[t].assign(static_cast<size_t>(H.num_blocks(t)), false);
      coupled_block_offset_[t].assign(static_cast<size_t>(H.num_blocks(t)), -1);
    }

    // Step 1: Detect off-diagonal couplings among eliminated variables
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
      constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
      const Index row_offset = H.base_offset(row_type);
      const Index col_offset = H.base_offset(col_type);
      const auto& indices = H.template block_indices<block_rows, block_cols>();
      structure_[K].assign(indices.begin(), indices.end());

      if (num_reduced_ == 0) return;

      if (row_offset >= num_reduced_ && col_offset >= num_reduced_) {
        if constexpr (row_type == col_type) {
          for (const auto& index : indices) {
            if (index.row != index.col) {
              is_block_coupled_[row_type][static_cast<size_t>(index.row)] = true;
              is_block_coupled_[row_type][static_cast<size_t>(index.col)] = true;
            }
          }
        } else {
          for (const auto& index : indices) {
            is_block_coupled_[row_type][static_cast<size_t>(index.row)] = true;
            is_block_coupled_[col_type][static_cast<size_t>(index.col)] = true;
          }
        }
      } else if (row_offset < num_reduced_ && col_offset >= num_reduced_) {
        auto& by_eliminated_block = coupling_blocks_[K];
        by_eliminated_block.resize(static_cast<size_t>(H.num_blocks(col_type)));
        for (size_t k = 0; k < indices.size(); ++k) {
          by_eliminated_block[static_cast<size_t>(indices[k].col)].push_back({indices[k].row, k});
        }
        for (auto& list : by_eliminated_block) {
          std::sort(list.begin(), list.end(),
                    [](const auto& a, const auto& b) { return a.row_block < b.row_block; });
        }
      }
    });

    if (num_reduced_ == 0) {
      if (!AnalyzeFullMatrix(H)) {
        info_ = Eigen::InvalidInput;
        return *this;
      }
      full_ldlt_.analyzePattern(full_sparse_);
      analyzed_ = (full_ldlt_.info() == Eigen::Success);
      return *this;
    }

    // Step 2: Partition eliminated blocks into independent blocks (I) and coupled blocks (C)
    Index num_coupled_dims = 0;
    Index num_independent_blocks = 0;
    for (size_t t = 0; t < kNumBlockTypes; ++t) {
      if (H.base_offset(t) < num_reduced_) continue;
      const int block_size = BlockMatrix::kBlockSizes[t];
      const size_t n_blocks = static_cast<size_t>(H.num_blocks(t));
      for (size_t b = 0; b < n_blocks; ++b) {
        if (is_block_coupled_[t][b]) {
          coupled_block_offset_[t][b] = num_coupled_dims;
          num_coupled_dims += block_size;
        } else {
          ++num_independent_blocks;
        }
      }
    }

    num_coupled_dims_ = num_coupled_dims;
    num_independent_blocks_ = num_independent_blocks;
    is_block_diagonal_ = (num_coupled_dims_ == 0);

    schur_.resize(num_reduced_, num_reduced_);
    reduced_ldlt_ = Eigen::LDLT<DenseMatrixType, Eigen::Upper>(num_reduced_);

    if (num_independent_blocks_ > 0) {
      [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(eliminated_inverses_).resize(static_cast<size_t>(H.num_blocks(Is)))), ...);
      }(std::make_index_sequence<kNumBlockTypes>{});
      eliminated_rhs_scratch_.resize(num_eliminated_);
    }

    if (num_coupled_dims_ > 0) {
      if (!AnalyzeEliminatedCoupledMatrix(H)) {
        info_ = Eigen::InvalidInput;
        return *this;
      }
      elim_ldlt_.analyzePattern(elim_sparse_);
      coupling_matrix_.resize(num_coupled_dims_, num_reduced_);
      coupling_matrix_.setZero();
      dense_w_.resize(num_coupled_dims_, num_reduced_);
      coupled_rhs_scratch_.resize(num_coupled_dims_);
      coupled_sol_scratch_.resize(num_coupled_dims_);
      coupled_backsub_scratch_.resize(num_coupled_dims_);
    }

    analyzed_ = true;
    return *this;
  }

  /// Numerical computation: factors the system without any dynamic memory allocations.
  BlockSparseLDLT& compute(const BlockMatrix& H) {
    computed_ = false;
    info_ = Eigen::Success;

    if (!analyzed_ || H.rows() != dims_ || H.cols() != dims_ || !MatchesStructure(H)) {
      info_ = Eigen::InvalidInput;
      return *this;
    }

    if (num_reduced_ == 0) {
      if (!UpdateFullMatrix(H)) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }
      full_ldlt_.factorize(full_sparse_);
      info_ = full_ldlt_.info();
      computed_ = (info_ == Eigen::Success);
      return *this;
    }

    schur_.setZero();
    bool finite_blocks = true;

    // Assemble the upper-left (reduced) portion into schur_
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
      constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
      const Index row_offset = H.base_offset(row_type);
      const Index col_offset = H.base_offset(col_type);
      if (row_offset >= num_reduced_ || col_offset >= num_reduced_) return;

      const auto& indices = H.template block_indices<block_rows, block_cols>();
      const auto& blocks = H.template blocks<block_rows, block_cols>();
      for (size_t k = 0; k < blocks.size(); ++k) {
        const auto& block = blocks[k];
        if (!block.allFinite()) {
          finite_blocks = false;
          continue;
        }
        const Index row = row_offset + indices[k].row * block_rows;
        const Index col = col_offset + indices[k].col * block_cols;
        schur_.template block<block_rows, block_cols>(row, col) = block;
        if constexpr (row_type != col_type) {
          schur_.template block<block_cols, block_rows>(col, row) = block.transpose();
        } else {
          if (indices[k].row != indices[k].col) {
            schur_.template block<block_cols, block_rows>(col, row) = block.transpose();
          }
        }
      }
    });

    if (!finite_blocks) {
      info_ = Eigen::NumericalIssue;
      return *this;
    }

    // 1. Independent eliminated blocks: fast closed-form inversion & direct Schur accumulation
    if (num_independent_blocks_ > 0) {
      if (!InvertEliminatedBlocks(H)) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }

      auto accumulate_schur_pair = [&](auto row_pair) {
        constexpr size_t K1 = decltype(row_pair)::value;
        constexpr auto row_types = internal::KToPair<kNumBlockTypes>(K1);
        constexpr size_t reduced_type_1 = row_types.first;
        constexpr size_t eliminated_type = row_types.second;
        if constexpr (reduced_type_1 < eliminated_type) {
          auto accumulate_column_pair = [&](auto column_pair) {
            constexpr size_t K2 = decltype(column_pair)::value;
            constexpr auto column_types = internal::KToPair<kNumBlockTypes>(K2);
            constexpr size_t reduced_type_2 = column_types.first;
            if constexpr (reduced_type_2 < eliminated_type && reduced_type_1 <= reduced_type_2 &&
                          column_types.second == eliminated_type) {
              constexpr int rows_1 = BlockMatrix::kBlockSizes[reduced_type_1];
              constexpr int rows_2 = BlockMatrix::kBlockSizes[reduced_type_2];
              constexpr int eliminated_size = BlockMatrix::kBlockSizes[eliminated_type];
              using EliminatedBlock = Matrix<Scalar, eliminated_size, eliminated_size>;
              const auto& blocks_1 = H.template blocks<rows_1, eliminated_size>();
              const auto& blocks_2 = H.template blocks<rows_2, eliminated_size>();
              const auto& lists_1 = coupling_blocks_[K1];
              const auto& lists_2 = coupling_blocks_[K2];
              const Index reduced_offset_1 = H.base_offset(reduced_type_1);
              const Index reduced_offset_2 = H.base_offset(reduced_type_2);
              if (reduced_offset_1 >= num_reduced_ || reduced_offset_2 >= num_reduced_) return;

              const auto& inv_vec = std::get<eliminated_type>(eliminated_inverses_);
              for (size_t eliminated_block = 0; eliminated_block < lists_1.size();
                   ++eliminated_block) {
                if (is_block_coupled_[eliminated_type][eliminated_block]) continue;

                const EliminatedBlock& inverse = inv_vec[eliminated_block];
                const auto& list_1 = lists_1[eliminated_block];
                const auto& list_2 = lists_2[eliminated_block];
                const size_t n1 = list_1.size();
                const size_t n2 = list_2.size();
                for (size_t i = 0; i < n1; ++i) {
                  const auto& entry_1 = list_1[i];
                  const Matrix<Scalar, rows_1, eliminated_size> B1_inv =
                      blocks_1[entry_1.matrix_block] * inverse;
                  const Index row =
                      reduced_offset_1 + static_cast<Index>(entry_1.row_block) * rows_1;
                  const size_t j_start = (K1 == K2) ? i : 0;
                  for (size_t j = j_start; j < n2; ++j) {
                    const auto& entry_2 = list_2[j];
                    const Index col =
                        reduced_offset_2 + static_cast<Index>(entry_2.row_block) * rows_2;
                    schur_.template block<rows_1, rows_2>(row, col).noalias() -=
                        B1_inv * blocks_2[entry_2.matrix_block].transpose();
                  }
                }
              }
            }
          };
          VisitBlockPairs(accumulate_column_pair);
        }
      };
      VisitBlockPairs(accumulate_schur_pair);
    }

    // 2. Coupled eliminated blocks: small sparse factorization & coupling solve
    if (num_coupled_dims_ > 0) {
      if (!UpdateEliminatedCoupledMatrix(H)) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }
      elim_ldlt_.factorize(elim_sparse_);
      if (elim_ldlt_.info() != Eigen::Success) {
        info_ = elim_ldlt_.info();
        return *this;
      }

      coupling_matrix_.setZero();
      VisitBlockPairs([&](auto pair_index) {
        constexpr size_t K = decltype(pair_index)::value;
        constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
        constexpr size_t row_type = types.first;
        constexpr size_t col_type = types.second;
        constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
        constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
        const Index row_offset = H.base_offset(row_type);
        const Index col_offset = H.base_offset(col_type);
        if (row_offset < num_reduced_ && col_offset >= num_reduced_) {
          const auto& indices = H.template block_indices<block_rows, block_cols>();
          const auto& blocks = H.template blocks<block_rows, block_cols>();
          for (size_t k = 0; k < blocks.size(); ++k) {
            const Index col_b = indices[k].col;
            if (!is_block_coupled_[col_type][static_cast<size_t>(col_b)]) continue;
            const Index r = row_offset + indices[k].row * block_rows;
            const Index c = coupled_block_offset_[col_type][static_cast<size_t>(col_b)];
            coupling_matrix_.template block<block_cols, block_rows>(c, r) = blocks[k].transpose();
          }
        }
      });

      dense_w_ = elim_ldlt_.solve(coupling_matrix_);
      if (!dense_w_.allFinite()) {
        info_ = Eigen::NumericalIssue;
        return *this;
      }

      schur_.noalias() -= coupling_matrix_.transpose() * dense_w_;
    }

    CompleteSymmetricMatrix(schur_);
    if (!schur_.allFinite()) {
      info_ = Eigen::NumericalIssue;
      return *this;
    }

    reduced_ldlt_.compute(schur_);
    info_ = reduced_ldlt_.info();
    computed_ = (info_ == Eigen::Success);
    if (computed_) {
      CacheMatrixData(H);
    }
    return *this;
  }

  /// Solves Ax = b for a vector right-hand side.
  [[nodiscard]] DenseVectorType solve(const DenseVectorType& rhs) const {
    DenseVectorType solution(dims_);
    if (!computed_ || rhs.size() != dims_ || !rhs.allFinite()) {
      solution.setConstant(std::numeric_limits<Scalar>::quiet_NaN());
      return solution;
    }

    if (num_reduced_ == 0) {
      solution = full_ldlt_.solve(rhs);
      return solution;
    }

    solvePartitionedVector(rhs, solution);
    return solution;
  }

  /// Solves AX = B for an N x M matrix right-hand side (e.g. B = I to compute full or partial
  /// inverse).
  [[nodiscard]] DenseMatrixType solve(const DenseMatrixType& B) const {
    DenseMatrixType X(dims_, B.cols());
    if (!computed_ || B.rows() != dims_ || !B.allFinite()) {
      X.setConstant(std::numeric_limits<Scalar>::quiet_NaN());
      return X;
    }

    if (num_reduced_ == 0) {
      X = full_ldlt_.solve(B);
      return X;
    }

    for (Index col = 0; col < B.cols(); ++col) {
      DenseVectorType x_col(dims_);
      solvePartitionedVector(B.col(col), x_col);
      X.col(col) = x_col;
    }

    return X;
  }

  /// Eigen-compatible generic MatrixBase solve overload.
  template <typename Derived>
  auto solve(const Eigen::MatrixBase<Derived>& b) const {
    if constexpr (Derived::ColsAtCompileTime == 1) {
      return solve(DenseVectorType(b));
    } else {
      return solve(DenseMatrixType(b));
    }
  }

  /// Tinyopt-style optional solver returning std::nullopt on failure.
  template <typename Rhs>
  auto Solve(const Rhs& b) const -> std::optional<decltype(solve(b))> {
    if (!computed_ || info_ != Eigen::Success) return std::nullopt;
    auto sol = solve(b);
    if (!sol.allFinite()) return std::nullopt;
    return sol;
  }

  [[nodiscard]] Eigen::ComputationInfo info() const noexcept { return info_; }
  [[nodiscard]] bool isAnalyzed() const noexcept { return analyzed_; }
  [[nodiscard]] bool isComputed() const noexcept { return computed_; }
  [[nodiscard]] bool isBlockDiagonalEliminated() const noexcept { return is_block_diagonal_; }
  [[nodiscard]] Index numCoupledDimensions() const noexcept { return num_coupled_dims_; }
  [[nodiscard]] Index numIndependentBlocks() const noexcept { return num_independent_blocks_; }
  [[nodiscard]] bool isPartitioned() const noexcept {
    return num_independent_blocks_ > 0 && num_coupled_dims_ > 0;
  }
  [[nodiscard]] Index rows() const noexcept { return dims_; }
  [[nodiscard]] Index cols() const noexcept { return dims_; }
  [[nodiscard]] Index numReduced() const noexcept { return num_reduced_; }
  [[nodiscard]] Index numEliminated() const noexcept { return num_eliminated_; }
  [[nodiscard]] const DenseMatrixType& schurMatrix() const noexcept { return schur_; }
  [[nodiscard]] bool matchesStructure(const BlockMatrix& H) const { return MatchesStructure(H); }

 private:
  template <typename Function>
  static void VisitBlockPairs(Function&& function) {
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (function(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumBlockPairs>{});
  }

  static bool IsGroupBoundary(const BlockMatrix& H, Index dimension) {
    for (size_t type = 0; type < kNumBlockTypes; ++type) {
      if (H.base_offset(type) == dimension) return true;
    }
    return false;
  }

  static void CompleteSymmetricMatrix(DenseMatrixType& matrix) {
    matrix.template triangularView<Eigen::Lower>() =
        matrix.template triangularView<Eigen::Upper>().transpose();
  }

  template <typename Block>
  static bool InvertAndStore(const Block& block, Block& inverse) {
    if (!block.allFinite()) return false;
    if constexpr (Block::RowsAtCompileTime > 0 && Block::RowsAtCompileTime <= 4) {
      inverse = block.inverse();
      if (inverse.allFinite()) return true;
    }
    const auto factor = block.template selfadjointView<Eigen::Upper>().ldlt();
    if (factor.info() != Eigen::Success) return false;
    const auto diagonal = factor.vectorD();
    if (!diagonal.allFinite() ||
        diagonal.minCoeff() <= Eigen::NumTraits<Scalar>::epsilon() * diagonal.maxCoeff()) {
      return false;
    }
    inverse = factor.solve(Block::Identity());
    if (!inverse.allFinite()) return false;
    return true;
  }

  bool InvertEliminatedBlocks(const BlockMatrix& H) {
    bool valid = true;
    auto invert_blocks = [&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t type = types.first;
      constexpr int block_size = BlockMatrix::kBlockSizes[type];
      if constexpr (types.first == types.second) {
        const Index block_offset = H.base_offset(type);
        if (block_offset < num_reduced_) return;
        using Block = Matrix<Scalar, block_size, block_size>;
        auto& inv_vec = std::get<type>(eliminated_inverses_);
        const auto& indices = H.template block_indices<block_size, block_size>();
        const auto& blocks = H.template blocks<block_size, block_size>();
        for (size_t k = 0; k < blocks.size(); ++k) {
          if (indices[k].row == indices[k].col) {
            const Index block = indices[k].row;
            if (is_block_coupled_[type][static_cast<size_t>(block)]) continue;
            Block inverse;
            if (!InvertAndStore(blocks[k], inverse)) {
              valid = false;
            } else {
              inv_vec[static_cast<size_t>(block)] = inverse;
            }
          }
        }
      }
    };
    VisitBlockPairs(invert_blocks);
    return valid;
  }

  bool MatchesStructure(const BlockMatrix& H) const {
    bool matches = true;
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr int block_rows = BlockMatrix::kBlockSizes[types.first];
      constexpr int block_cols = BlockMatrix::kBlockSizes[types.second];
      const auto& indices = H.template block_indices<block_rows, block_cols>();
      if (indices.size() != structure_[K].size() || indices != structure_[K]) matches = false;
    });
    return matches;
  }

  void solvePartitionedVector(const DenseVectorType& rhs, DenseVectorType& solution) const {
    DenseVectorType reduced_rhs = rhs.head(num_reduced_);

    // 1. Accumulate RHS from independent eliminated variables
    if (num_independent_blocks_ > 0) {
      auto accumulate_rhs = [&](auto pair_index) {
        constexpr size_t K = decltype(pair_index)::value;
        constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
        constexpr size_t reduced_type = types.first;
        constexpr size_t eliminated_type = types.second;
        if constexpr (reduced_type < eliminated_type) {
          constexpr int reduced_size = BlockMatrix::kBlockSizes[reduced_type];
          constexpr int eliminated_size = BlockMatrix::kBlockSizes[eliminated_type];
          using EliminatedBlock = Matrix<Scalar, eliminated_size, eliminated_size>;
          using EliminatedVector = Vector<Scalar, eliminated_size>;
          const auto& blocks = *std::get<K>(cached_blocks_);
          const auto& lists = coupling_blocks_[K];
          const Index reduced_offset = cached_base_offsets_[reduced_type];
          const Index eliminated_offset = cached_base_offsets_[eliminated_type];
          const auto& inv_vec = std::get<eliminated_type>(eliminated_inverses_);
          for (size_t eliminated_block = 0; eliminated_block < lists.size(); ++eliminated_block) {
            if (is_block_coupled_[eliminated_type][eliminated_block]) continue;
            const EliminatedBlock& inverse = inv_vec[eliminated_block];
            const EliminatedVector eliminated_rhs_block = rhs.template segment<eliminated_size>(
                eliminated_offset + static_cast<Index>(eliminated_block) * eliminated_size);
            const EliminatedVector solved_rhs = inverse * eliminated_rhs_block;
            for (const CouplingBlock& entry : lists[eliminated_block]) {
              const Index row = reduced_offset + static_cast<Index>(entry.row_block) * reduced_size;
              reduced_rhs.template segment<reduced_size>(row).noalias() -=
                  blocks[entry.matrix_block] * solved_rhs;
            }
          }
        }
      };
      VisitBlockPairs(accumulate_rhs);
    }

    // 2. Accumulate RHS from coupled eliminated variables
    if (num_coupled_dims_ > 0) {
      for (size_t t = 0; t < kNumBlockTypes; ++t) {
        if (cached_base_offsets_[t] < num_reduced_) continue;
        const int block_size = cached_block_sizes_[t];
        for (size_t b = 0; b < is_block_coupled_[t].size(); ++b) {
          if (is_block_coupled_[t][b]) {
            const Index c_off = coupled_block_offset_[t][b];
            const Index r_off = cached_base_offsets_[t] + static_cast<Index>(b) * block_size;
            coupled_rhs_scratch_.segment(c_off, block_size) = rhs.segment(r_off, block_size);
          }
        }
      }
      coupled_sol_scratch_ = elim_ldlt_.solve(coupled_rhs_scratch_);
      reduced_rhs.noalias() -= coupling_matrix_.transpose() * coupled_sol_scratch_;
    }

    // 3. Solve reduced system
    solution.head(num_reduced_) = reduced_ldlt_.solve(reduced_rhs);

    // 4. Back-substitute independent eliminated variables
    if (num_independent_blocks_ > 0) {
      eliminated_rhs_scratch_ = rhs.tail(num_eliminated_);
      auto back_substitute_coupling = [&](auto pair_index) {
        constexpr size_t K = decltype(pair_index)::value;
        constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
        constexpr size_t reduced_type = types.first;
        constexpr size_t eliminated_type = types.second;
        if constexpr (reduced_type < eliminated_type) {
          constexpr int reduced_size = BlockMatrix::kBlockSizes[reduced_type];
          constexpr int eliminated_size = BlockMatrix::kBlockSizes[eliminated_type];
          const auto& blocks = *std::get<K>(cached_blocks_);
          const auto& lists = coupling_blocks_[K];
          const Index reduced_offset = cached_base_offsets_[reduced_type];
          const Index eliminated_offset = cached_base_offsets_[eliminated_type] - num_reduced_;
          for (size_t eliminated_block = 0; eliminated_block < lists.size(); ++eliminated_block) {
            if (is_block_coupled_[eliminated_type][eliminated_block]) continue;
            auto eliminated_block_rhs = eliminated_rhs_scratch_.template segment<eliminated_size>(
                eliminated_offset + eliminated_block * eliminated_size);
            for (const CouplingBlock& entry : lists[eliminated_block]) {
              const Index row = reduced_offset + static_cast<Index>(entry.row_block) * reduced_size;
              eliminated_block_rhs.noalias() -= blocks[entry.matrix_block].transpose() *
                                                solution.template segment<reduced_size>(row);
            }
          }
        }
      };
      VisitBlockPairs(back_substitute_coupling);

      auto solve_eliminated_blocks = [&](auto pair_index) {
        constexpr size_t K = decltype(pair_index)::value;
        constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
        constexpr size_t type = types.first;
        constexpr int block_size = BlockMatrix::kBlockSizes[type];
        if constexpr (types.first == types.second) {
          const Index block_offset = cached_base_offsets_[type];
          if (block_offset < num_reduced_) return;
          const auto& inv_vec = std::get<type>(eliminated_inverses_);
          const size_t n_blocks = inv_vec.size();
          const Index offset_base = block_offset - num_reduced_;
          const Index sol_offset_base = block_offset;
          for (size_t block = 0; block < n_blocks; ++block) {
            if (is_block_coupled_[type][block]) continue;
            const Index off = offset_base + static_cast<Index>(block) * block_size;
            const Index sol_off = sol_offset_base + static_cast<Index>(block) * block_size;
            solution.template segment<block_size>(sol_off).noalias() =
                inv_vec[block] * eliminated_rhs_scratch_.template segment<block_size>(off);
          }
        }
      };
      VisitBlockPairs(solve_eliminated_blocks);
    }

    // 5. Back-substitute coupled eliminated variables
    if (num_coupled_dims_ > 0) {
      coupled_backsub_scratch_ =
          coupled_rhs_scratch_ - coupling_matrix_ * solution.head(num_reduced_);
      coupled_sol_scratch_ = elim_ldlt_.solve(coupled_backsub_scratch_);

      for (size_t t = 0; t < kNumBlockTypes; ++t) {
        if (cached_base_offsets_[t] < num_reduced_) continue;
        const int block_size = cached_block_sizes_[t];
        for (size_t b = 0; b < is_block_coupled_[t].size(); ++b) {
          if (is_block_coupled_[t][b]) {
            const Index c_off = coupled_block_offset_[t][b];
            const Index s_off = cached_base_offsets_[t] + static_cast<Index>(b) * block_size;
            solution.segment(s_off, block_size) = coupled_sol_scratch_.segment(c_off, block_size);
          }
        }
      }
    }
  }

  bool AnalyzeFullMatrix(const BlockMatrix& H) {
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(static_cast<size_t>(H.nonZeros()));
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
      constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
      const Index row_offset = H.base_offset(row_type);
      const Index col_offset = H.base_offset(col_type);
      const auto& indices = H.template block_indices<block_rows, block_cols>();
      for (size_t k = 0; k < indices.size(); ++k) {
        const Index row = row_offset + indices[k].row * block_rows;
        const Index col = col_offset + indices[k].col * block_cols;
        const bool has_symmetric_mate = row_type != col_type || indices[k].row != indices[k].col;
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            triplets.emplace_back(row + r, col + c, Scalar(0));
            if (has_symmetric_mate) triplets.emplace_back(col + c, row + r, Scalar(0));
          }
        }
      }
    });
    full_sparse_.resize(dims_, dims_);
    full_sparse_.setFromTriplets(triplets.begin(), triplets.end());
    full_sparse_.makeCompressed();

    bool structure_valid = true;
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
      constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
      const Index row_offset = H.base_offset(row_type);
      const Index col_offset = H.base_offset(col_type);
      const auto& indices = H.template block_indices<block_rows, block_cols>();
      auto& slots = full_value_slots_[K];
      slots.resize(indices.size() * static_cast<size_t>(block_rows * block_cols));
      for (size_t k = 0; k < indices.size(); ++k) {
        const Index row = row_offset + indices[k].row * block_rows;
        const Index col = col_offset + indices[k].col * block_cols;
        const bool has_symmetric_mate = row_type != col_type || indices[k].row != indices[k].col;
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            const size_t slot_index =
                (k * static_cast<size_t>(block_rows) + static_cast<size_t>(r)) *
                    static_cast<size_t>(block_cols) +
                static_cast<size_t>(c);
            auto& slot = slots[slot_index];
            slot.first = FindValueIndex(full_sparse_, row + r, col + c);
            if (slot.first < 0) {
              structure_valid = false;
              continue;
            }
            if (has_symmetric_mate) {
              slot.second = FindValueIndex(full_sparse_, col + c, row + r);
              if (slot.second < 0) structure_valid = false;
            }
          }
        }
      }
    });
    return structure_valid;
  }

  bool UpdateFullMatrix(const BlockMatrix& H) {
    auto* values = full_sparse_.valuePtr();
    bool finite_blocks = true;
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
      constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
      const auto& blocks = H.template blocks<block_rows, block_cols>();
      const auto& slots = full_value_slots_[K];
      for (size_t k = 0; k < blocks.size(); ++k) {
        const auto& block = blocks[k];
        if (!block.allFinite()) {
          finite_blocks = false;
          continue;
        }
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            const size_t slot_index =
                (k * static_cast<size_t>(block_rows) + static_cast<size_t>(r)) *
                    static_cast<size_t>(block_cols) +
                static_cast<size_t>(c);
            const auto& slot = slots[slot_index];
            values[slot.first] = block(r, c);
            if (slot.second >= 0) values[slot.second] = block(r, c);
          }
        }
      }
    });
    return finite_blocks;
  }

  bool AnalyzeEliminatedCoupledMatrix(const BlockMatrix& H) {
    std::vector<Eigen::Triplet<Scalar>> triplets;
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
      constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
      const Index row_offset = H.base_offset(row_type);
      const Index col_offset = H.base_offset(col_type);
      if (row_offset < num_reduced_ || col_offset < num_reduced_) return;

      const auto& indices = H.template block_indices<block_rows, block_cols>();
      for (size_t k = 0; k < indices.size(); ++k) {
        const Index row_b = indices[k].row;
        const Index col_b = indices[k].col;
        if (!is_block_coupled_[row_type][static_cast<size_t>(row_b)] ||
            !is_block_coupled_[col_type][static_cast<size_t>(col_b)]) {
          continue;
        }

        const Index c_row = coupled_block_offset_[row_type][static_cast<size_t>(row_b)];
        const Index c_col = coupled_block_offset_[col_type][static_cast<size_t>(col_b)];
        const bool has_symmetric_mate = row_type != col_type || row_b != col_b;
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            triplets.emplace_back(c_row + r, c_col + c, Scalar(0));
            if (has_symmetric_mate) triplets.emplace_back(c_col + c, c_row + r, Scalar(0));
          }
        }
      }
    });

    elim_sparse_.resize(num_coupled_dims_, num_coupled_dims_);
    elim_sparse_.setFromTriplets(triplets.begin(), triplets.end());
    elim_sparse_.makeCompressed();

    bool structure_valid = true;
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
      constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
      const Index row_offset = H.base_offset(row_type);
      const Index col_offset = H.base_offset(col_type);
      if (row_offset < num_reduced_ || col_offset < num_reduced_) return;

      const auto& indices = H.template block_indices<block_rows, block_cols>();
      auto& slots = elim_value_slots_[K];
      slots.resize(indices.size() * static_cast<size_t>(block_rows * block_cols));
      for (size_t k = 0; k < indices.size(); ++k) {
        const Index row_b = indices[k].row;
        const Index col_b = indices[k].col;
        const bool is_coupled = is_block_coupled_[row_type][static_cast<size_t>(row_b)] &&
                                is_block_coupled_[col_type][static_cast<size_t>(col_b)];
        const Index c_row =
            is_coupled ? coupled_block_offset_[row_type][static_cast<size_t>(row_b)] : -1;
        const Index c_col =
            is_coupled ? coupled_block_offset_[col_type][static_cast<size_t>(col_b)] : -1;
        const bool has_symmetric_mate = row_type != col_type || row_b != col_b;
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            const size_t slot_index =
                (k * static_cast<size_t>(block_rows) + static_cast<size_t>(r)) *
                    static_cast<size_t>(block_cols) +
                static_cast<size_t>(c);
            auto& slot = slots[slot_index];
            if (!is_coupled) {
              slot.first = -1;
              slot.second = -1;
              continue;
            }
            slot.first = FindValueIndex(elim_sparse_, c_row + r, c_col + c);
            if (slot.first < 0) {
              structure_valid = false;
              continue;
            }
            if (has_symmetric_mate) {
              slot.second = FindValueIndex(elim_sparse_, c_col + c, c_row + r);
              if (slot.second < 0) structure_valid = false;
            }
          }
        }
      }
    });
    return structure_valid;
  }

  bool UpdateEliminatedCoupledMatrix(const BlockMatrix& H) {
    auto* values = elim_sparse_.valuePtr();
    bool finite_blocks = true;
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
      constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
      const Index row_offset = H.base_offset(row_type);
      const Index col_offset = H.base_offset(col_type);
      if (row_offset < num_reduced_ || col_offset < num_reduced_) return;

      const auto& blocks = H.template blocks<block_rows, block_cols>();
      const auto& slots = elim_value_slots_[K];
      for (size_t k = 0; k < blocks.size(); ++k) {
        const auto& block = blocks[k];
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            const size_t slot_index =
                (k * static_cast<size_t>(block_rows) + static_cast<size_t>(r)) *
                    static_cast<size_t>(block_cols) +
                static_cast<size_t>(c);
            const auto& slot = slots[slot_index];
            if (slot.first < 0) continue;
            if (!std::isfinite(block(r, c))) {
              finite_blocks = false;
              continue;
            }
            values[slot.first] = block(r, c);
            if (slot.second >= 0) values[slot.second] = block(r, c);
          }
        }
      }
    });
    return finite_blocks;
  }

  static Index FindValueIndex(const SparseMatrixType& mat, Index row, Index col) {
    const auto* outer = mat.outerIndexPtr();
    const auto* inner = mat.innerIndexPtr();
    const auto* begin = inner + outer[col];
    const auto* end = inner + outer[col + 1];
    const auto* found = std::lower_bound(begin, end, row);
    if (found == end || *found != row) return -1;
    return static_cast<Index>(found - inner);
  }

  // Cached pointers/offsets for zero-copy solve access
  template <size_t... Ks>
  static auto MakeCachedBlocksTuple(std::index_sequence<Ks...>)
      -> std::tuple<const std::vector<
          Matrix<Scalar, BlockMatrix::kBlockSizes[internal::KToPair<kNumBlockTypes>(Ks).first],
                 BlockMatrix::kBlockSizes[internal::KToPair<kNumBlockTypes>(Ks).second]>>*...>;

  using CachedBlocksTupleType =
      decltype(MakeCachedBlocksTuple(std::make_index_sequence<kNumBlockPairs>{}));

  void CacheMatrixData(const BlockMatrix& H) {
    for (size_t i = 0; i < kNumBlockTypes; ++i) {
      cached_base_offsets_[i] = H.base_offset(i);
      cached_block_sizes_[i] = BlockMatrix::kBlockSizes[i];
    }
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr int block_rows = BlockMatrix::kBlockSizes[types.first];
      constexpr int block_cols = BlockMatrix::kBlockSizes[types.second];
      std::get<K>(cached_blocks_) = &H.template blocks<block_rows, block_cols>();
    });
  }

  Index dims_ = 0;
  Index num_reduced_ = 0;
  Index num_eliminated_ = 0;
  bool analyzed_ = false;
  bool computed_ = false;
  bool is_block_diagonal_ = true;
  Eigen::ComputationInfo info_ = Eigen::Success;

  // Partitioned eliminated block structures
  Index num_coupled_dims_ = 0;
  Index num_independent_blocks_ = 0;
  std::array<std::vector<bool>, kNumBlockTypes> is_block_coupled_;
  std::array<std::vector<Index>, kNumBlockTypes> coupled_block_offset_;

  // Block diagonal / independent structures
  DenseMatrixType schur_;
  Eigen::LDLT<DenseMatrixType, Eigen::Upper> reduced_ldlt_;
  InversesTupleType eliminated_inverses_;
  std::array<std::vector<std::vector<CouplingBlock>>, kNumBlockPairs> coupling_blocks_;
  std::array<Index, kNumBlockTypes> cached_base_offsets_{};
  std::array<int, kNumBlockTypes> cached_block_sizes_{};
  CachedBlocksTupleType cached_blocks_;
  mutable DenseVectorType eliminated_rhs_scratch_;

  // Coupled eliminated structures
  SparseMatrixType elim_sparse_;
  Eigen::SimplicialLDLT<SparseMatrixType, Eigen::Upper, Eigen::AMDOrdering<int>> elim_ldlt_;
  DenseMatrixType coupling_matrix_;
  DenseMatrixType dense_w_;
  std::array<std::vector<MatrixValueSlot>, kNumBlockPairs> elim_value_slots_;
  mutable DenseVectorType coupled_rhs_scratch_;
  mutable DenseVectorType coupled_sol_scratch_;
  mutable DenseVectorType coupled_backsub_scratch_;

  // Full matrix factorization structures
  SparseMatrixType full_sparse_;
  Eigen::SimplicialLDLT<SparseMatrixType, Eigen::Upper, Eigen::AMDOrdering<int>> full_ldlt_;
  std::array<std::vector<MatrixValueSlot>, kNumBlockPairs> full_value_slots_;

  // Structure tracking for pattern change validation
  std::array<std::vector<BlockIndex>, kNumBlockPairs> structure_;
};

}  // namespace tinyopt
