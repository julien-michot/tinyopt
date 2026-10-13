// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/math/solvers/base.h>
#include <tinyopt/math/solvers/dense.h>
#include <tinyopt/math/solvers/sparse.h>
#include <tinyopt/types.h>

namespace tinyopt {

template <typename BlockMatrix>
struct BlockSparseLinearSolvers {
  using Scalar = typename BlockMatrix::ScalarType;
  using SparseMatrixType = SparseMatrix<Scalar>;
  using DenseMatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseSolver = DenseLinearSolvers<DenseMatrixType>;
  static constexpr size_t kNumBlockTypes = BlockMatrix::kNumBlockTypes;
  static constexpr size_t kNumBlockPairs = BlockMatrix::kNumPairs;

  struct EliminatedValueSlot {
    Index first = -1;
    Index second = -1;
  };

  Index dims = 0;
  Index num_reduced_ = 0;
  LinearSolverOptions options{};
  SparseMatrixType eliminated_matrix;
  SparseLinearSolvers<SparseMatrixType> eliminated_solver;
  DenseSolver reduced_solver;

  explicit BlockSparseLinearSolvers(const LinearSolverOptions& solver_options = {})
      : options(solver_options),
        eliminated_solver(SparseOptions(solver_options), 0),
        reduced_solver(ReducedOptions(solver_options), 0) {}

  void reset() {
    analyzed_ = false;
    computed_ = false;
    num_reduced_ = 0;
    eliminated_solver.reset();
    reduced_solver.reset();
  }

  void resize(Index new_dims, Index num_reduced = 0) {
    if (dims != new_dims || num_reduced_ != num_reduced) {
      dims = new_dims;
      reset();
    }
    num_reduced_ = num_reduced;
    if (num_reduced > 0) reduced_solver.resize(num_reduced);
  }

  /// Build the fixed pattern; a zero reduced dimension factors the complete system directly.
  bool Analyze(const BlockMatrix& H, Index num_reduced = 0) {
    analyzed_ = false;
    computed_ = false;
    const Index dimension = H.rows();
    if (H.cols() != dimension || dimension <= 0 || num_reduced < 0 || num_reduced >= dimension ||
        (num_reduced > 0 && !options.allow_schur) || !IsGroupBoundary(H, num_reduced)) {
      num_reduced_ = -1;
      analyzed_ = false;
      computed_ = false;
      return false;
    }

    dims = dimension;
    num_reduced_ = num_reduced;
    num_eliminated_ = dimension - num_reduced;
    schur_.resize(num_reduced_, num_reduced_);
    coupling_.resize(num_reduced_, num_eliminated_);
    solved_coupling_.resize(num_eliminated_, num_reduced_);

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
      const auto& blocks = H.template blocks<block_rows, block_cols>();

      block_counts_[K] = blocks.size();
      structure_[K].assign(indices.begin(), indices.end());
      const bool eliminated_pair = row_offset >= num_reduced_ && col_offset >= num_reduced_;
      if (!eliminated_pair) return;

      for (size_t k = 0; k < blocks.size(); ++k) {
        const Index row = row_offset + indices[k].row * block_rows - num_reduced_;
        const Index col = col_offset + indices[k].col * block_cols - num_reduced_;
        const bool has_symmetric_mate = row_type != col_type || indices[k].row != indices[k].col;
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            triplets.emplace_back(row + r, col + c, Scalar(0));
            if (has_symmetric_mate) {
              triplets.emplace_back(col + c, row + r, Scalar(0));
            }
          }
        }
      }
    });
    eliminated_matrix.resize(num_eliminated_, num_eliminated_);
    eliminated_matrix.setFromTriplets(triplets.begin(), triplets.end());
    eliminated_matrix.makeCompressed();

    for (auto& slots : eliminated_value_slots_) slots.clear();
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
      auto& slots = eliminated_value_slots_[K];
      slots.resize(indices.size() * static_cast<size_t>(block_rows * block_cols));
      if (row_offset < num_reduced_ || col_offset < num_reduced_) return;

      for (size_t k = 0; k < indices.size(); ++k) {
        const Index row = row_offset + indices[k].row * block_rows - num_reduced_;
        const Index col = col_offset + indices[k].col * block_cols - num_reduced_;
        const bool has_symmetric_mate = row_type != col_type || indices[k].row != indices[k].col;
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            const size_t slot_index =
                (k * static_cast<size_t>(block_rows) + static_cast<size_t>(r)) *
                    static_cast<size_t>(block_cols) +
                static_cast<size_t>(c);
            auto& slot = slots[slot_index];
            slot.first = FindValueIndex(row + r, col + c);
            if (slot.first < 0) {
              structure_valid = false;
              continue;
            }
            if (has_symmetric_mate) {
              slot.second = FindValueIndex(col + c, row + r);
              if (slot.second < 0) structure_valid = false;
            }
          }
        }
      }
    });
    if (!structure_valid) return false;

    eliminated_solver.resize(num_eliminated_);
    eliminated_solver.Analyze(eliminated_matrix);
    if (!eliminated_solver.IsAnalyzed()) return false;

    if (num_reduced_ > 0) reduced_solver.resize(num_reduced_);
    analyzed_ = true;
    return true;
  }

  /// Refresh block values and numerically factor the eliminated and reduced systems.
  bool Compute(const BlockMatrix& H) {
    computed_ = false;
    if (!analyzed_ || H.rows() != dims || H.cols() != dims || !MatchesStructure(H)) {
      return false;
    }

    schur_.setZero();
    coupling_.setZero();
    auto* eliminated_values = eliminated_matrix.valuePtr();
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
      const bool row_reduced = row_offset < num_reduced_;
      const bool col_reduced = col_offset < num_reduced_;
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
        const bool has_symmetric_mate = row_type != col_type || indices[k].row != indices[k].col;

        if (row_reduced && col_reduced) {
          schur_.block(row, col, block_rows, block_cols) += block;
          if (has_symmetric_mate) {
            schur_.block(col, row, block_cols, block_rows) += block.transpose();
          }
        } else if (row_reduced) {
          coupling_.block(row, col - num_reduced_, block_rows, block_cols) = block;
        } else {
          const auto& slots = eliminated_value_slots_[K];
          for (int c = 0; c < block_cols; ++c) {
            for (int r = 0; r < block_rows; ++r) {
              const size_t slot_index =
                  (k * static_cast<size_t>(block_rows) + static_cast<size_t>(r)) *
                      static_cast<size_t>(block_cols) +
                  static_cast<size_t>(c);
              const auto& slot = slots[slot_index];
              eliminated_values[slot.first] = block(r, c);
              if (slot.second >= 0) eliminated_values[slot.second] = block(r, c);
            }
          }
        }
      }
    });
    if (!finite_blocks || !eliminated_solver.Compute(eliminated_matrix)) return false;

    if (num_reduced_ == 0) {
      computed_ = true;
      return true;
    }

    solved_coupling_ = eliminated_solver.SolverLinear(coupling_.transpose());
    if (eliminated_solver.Info() != Eigen::Success || !solved_coupling_.allFinite()) {
      return false;
    }

    auto accumulate_schur_update = [&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = types.first;
      constexpr size_t col_type = types.second;
      if constexpr (row_type < col_type) {
        constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
        constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];
        const Index row_offset = H.base_offset(row_type);
        const Index col_offset = H.base_offset(col_type);
        if (row_offset >= num_reduced_ || col_offset < num_reduced_) return;

        const auto& indices = H.template block_indices<block_rows, block_cols>();
        const auto& blocks = H.template blocks<block_rows, block_cols>();
        for (size_t k = 0; k < blocks.size(); ++k) {
          const Index row = row_offset + indices[k].row * block_rows;
          const Index col = col_offset + indices[k].col * block_cols - num_reduced_;
          schur_.block(row, 0, block_rows, num_reduced_).noalias() -=
              blocks[k] * solved_coupling_.block(col, 0, block_cols, num_reduced_);
        }
      }
    };
    VisitBlockPairs(accumulate_schur_update);

    CompleteSymmetricMatrix(schur_);
    if (!schur_.allFinite() || !reduced_solver.Compute(schur_)) return false;
    computed_ = true;
    return true;
  }

  /// Solve once, analyzing the matrix first if the structure or partition has changed.
  std::optional<Vector<Scalar, Dynamic>> Solve(const BlockMatrix& H,
                                               const Vector<Scalar, Dynamic>& rhs) {
    if (num_reduced_ < 0 || num_reduced_ >= H.rows() ||
        (num_reduced_ > 0 && !options.allow_schur)) {
      return std::nullopt;
    }
    if (!analyzed_ || !MatchesStructure(H)) {
      if (!Analyze(H, num_reduced_)) return std::nullopt;
    }
    if (!Compute(H)) return std::nullopt;
    return SolverLinear(rhs);
  }

  /// Solve using the factorization produced by the latest successful compute().
 private:
  std::optional<Vector<Scalar, Dynamic>> SolverLinear(const Vector<Scalar, Dynamic>& rhs) const {
    if (!computed_ || rhs.size() != dims || !rhs.allFinite()) return std::nullopt;

    if (num_reduced_ == 0) {
      const Vector<Scalar, Dynamic> solution = eliminated_solver.SolverLinear(rhs);
      if (eliminated_solver.Info() != Eigen::Success || !solution.allFinite()) {
        return std::nullopt;
      }
      return solution;
    }

    const auto rhs_eliminated = rhs.tail(num_eliminated_);
    const Vector<Scalar, Dynamic> solved_rhs = eliminated_solver.SolverLinear(rhs_eliminated);
    if (eliminated_solver.Info() != Eigen::Success || !solved_rhs.allFinite()) {
      return std::nullopt;
    }
    Vector<Scalar, Dynamic> reduced_rhs = rhs.head(num_reduced_);
    reduced_rhs.noalias() -= coupling_ * solved_rhs;
    const auto reduced_solution = reduced_solver.SolverLinear(reduced_rhs);
    if (!reduced_solution) return std::nullopt;

    Vector<Scalar, Dynamic> solution(dims);
    solution.head(num_reduced_) = *reduced_solution;
    solution.tail(num_eliminated_) = eliminated_solver.SolverLinear(
        rhs_eliminated - coupling_.transpose() * solution.head(num_reduced_));
    if (eliminated_solver.Info() != Eigen::Success || !solution.allFinite()) {
      return std::nullopt;
    }
    return solution;
  }

 public:
  [[nodiscard]] bool IsAnalyzed() const noexcept { return analyzed_; }
  [[nodiscard]] bool IsComputed() const noexcept { return computed_; }
  [[nodiscard]] LinearSolverMethod GetMethod() const { return options.method; }
  [[nodiscard]] Index Dims() const { return dims; }

 private:
  static LinearSolverOptions ReducedOptions(LinearSolverOptions solver_options) {
    if (solver_options.method == LinearSolverMethod::SuiteSparse ||
        solver_options.method == LinearSolverMethod::IncompleteCholesky) {
      solver_options.method = LinearSolverMethod::LDLT;
    }
    return solver_options;
  }

  static LinearSolverOptions SparseOptions(LinearSolverOptions solver_options) {
    if (solver_options.method == LinearSolverMethod::SuiteSparse) {
      solver_options.method = LinearSolverMethod::LDLT;
    }
    return solver_options;
  }

  static bool IsGroupBoundary(const BlockMatrix& H, Index dimension) {
    for (size_t type = 0; type < kNumBlockTypes; ++type) {
      if (H.base_offset(type) == dimension) return true;
    }
    return false;
  }

  template <typename Function>
  static void VisitBlockPairs(Function&& function) {
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (function(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumBlockPairs>{});
  }

  Index FindValueIndex(Index row, Index col) const {
    const auto* outer = eliminated_matrix.outerIndexPtr();
    const auto* inner = eliminated_matrix.innerIndexPtr();
    const auto* begin = inner + outer[col];
    const auto* end = inner + outer[col + 1];
    const auto* found = std::lower_bound(begin, end, row);
    if (found == end || *found != row) return -1;
    return static_cast<Index>(found - inner);
  }

  bool MatchesStructure(const BlockMatrix& H) const {
    bool matches = true;
    VisitBlockPairs([&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr int block_rows = BlockMatrix::kBlockSizes[types.first];
      constexpr int block_cols = BlockMatrix::kBlockSizes[types.second];
      const auto& indices = H.template block_indices<block_rows, block_cols>();
      if (indices.size() != block_counts_[K] || indices != structure_[K]) matches = false;
    });
    return matches;
  }

  Index num_eliminated_ = 0;
  DenseMatrixType schur_;
  DenseMatrixType coupling_;
  DenseMatrixType solved_coupling_;
  std::array<size_t, kNumBlockPairs> block_counts_{};
  std::array<std::vector<BlockIndex>, kNumBlockPairs> structure_;
  std::array<std::vector<EliminatedValueSlot>, kNumBlockPairs> eliminated_value_slots_;
  bool analyzed_ = false;
  bool computed_ = false;
};

}  // namespace tinyopt
