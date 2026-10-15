// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include <Eigen/Cholesky>

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

  struct CouplingBlock {
    Index row_block = 0;
    size_t matrix_block = 0;
  };

  struct MatrixValueSlot {
    Index first = -1;
    Index second = -1;
  };

  Index dims = 0;
  Index num_reduced_ = 0;
  LinearSolverOptions options{};
  SparseMatrixType eliminated_matrix;
  SparseLinearSolvers<SparseMatrixType> eliminated_solver;
  DenseSolver reduced_solver;

  template <size_t... Is>
  static auto MakeInversesTuple(std::index_sequence<Is...>)
      -> std::tuple<std::vector<
          Matrix<Scalar, BlockMatrix::kBlockSizes[Is], BlockMatrix::kBlockSizes[Is]>>...>;

  using InversesTupleType = decltype(MakeInversesTuple(std::make_index_sequence<kNumBlockTypes>{}));

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
    std::apply([](auto&... vecs) { (vecs.clear(), ...); }, eliminated_inverses_);
    for (auto& blocks : coupling_blocks_) blocks.clear();
  }

  void resize(Index new_dims, Index num_reduced = 0) {
    if (dims != new_dims || num_reduced_ != num_reduced) {
      dims = new_dims;
      reset();
    }
    num_reduced_ = num_reduced;
    if (num_reduced > 0) reduced_solver.resize(num_reduced);
  }

  /// Analyze a full sparse system or a Schur system with block-diagonal eliminated variables.
  bool Analyze(const BlockMatrix& H, Index num_reduced = 0) {
    analyzed_ = false;
    computed_ = false;
    eliminated_solver.reset();
    reduced_solver.reset();
    for (auto& structure : structure_) structure.clear();
    for (auto& blocks : coupling_blocks_) blocks.clear();
    for (auto& slots : full_value_slots_) slots.clear();
    std::apply([](auto&... vecs) { (vecs.clear(), ...); }, eliminated_inverses_);

    const Index dimension = H.rows();
    if (H.cols() != dimension || dimension <= 0 || num_reduced < 0 || num_reduced >= dimension ||
        (num_reduced > 0 && !options.allow_schur) || !IsGroupBoundary(H, num_reduced)) {
      num_reduced_ = -1;
      return false;
    }

    dims = dimension;
    num_reduced_ = num_reduced;
    num_eliminated_ = dimension - num_reduced;
    if (num_reduced_ > 0) {
      SparseMatrixType{}.swap(eliminated_matrix);
    }

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
      structure_[K].assign(indices.begin(), indices.end());

      if (num_reduced_ == 0) return;
      if (row_offset >= num_reduced_ && col_offset >= num_reduced_) {
        if constexpr (row_type == col_type) {
          for (const auto& index : indices) {
            if (index.row != index.col) structure_valid = false;
          }
        } else if (!indices.empty()) {
          structure_valid = false;
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
    if (!structure_valid) {
      num_reduced_ = -1;
      return false;
    }

    if (num_reduced_ == 0) {
      if (!AnalyzeFullMatrix(H)) return false;
      eliminated_solver.Analyze(eliminated_matrix);
      analyzed_ = eliminated_solver.IsAnalyzed();
      return analyzed_;
    }

    [&]<size_t... Is>(std::index_sequence<Is...>) {
      ((std::get<Is>(eliminated_inverses_).resize(static_cast<size_t>(H.num_blocks(Is)))), ...);
    }(std::make_index_sequence<kNumBlockTypes>{});
    schur_.resize(num_reduced_, num_reduced_);
    reduced_solver.resize(num_reduced_);
    analyzed_ = true;
    return true;
  }

  /// Refresh block values, invert each independent eliminated block and factor the Schur system.
  bool Compute(const BlockMatrix& H) {
    computed_ = false;
    if (!analyzed_ || H.rows() != dims || H.cols() != dims || !MatchesStructure(H)) return false;

    if (num_reduced_ == 0) {
      if (!UpdateFullMatrix(H)) return false;
      computed_ = eliminated_solver.Compute(eliminated_matrix);
      return computed_;
    }

    schur_.setZero();
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
              const EliminatedBlock& inverse = inv_vec[eliminated_block];
              const auto& list_1 = lists_1[eliminated_block];
              const auto& list_2 = lists_2[eliminated_block];
              const size_t n1 = list_1.size();
              const size_t n2 = list_2.size();
              for (size_t i = 0; i < n1; ++i) {
                const auto& entry_1 = list_1[i];
                const Matrix<Scalar, rows_1, eliminated_size> B1_inv =
                    blocks_1[entry_1.matrix_block] * inverse;
                const Index row = reduced_offset_1 + static_cast<Index>(entry_1.row_block) * rows_1;
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
    if (!finite_blocks || !InvertEliminatedBlocks(H)) return false;
    VisitBlockPairs(accumulate_schur_pair);
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
    return SolverLinear(H, rhs);
  }

 private:
  std::optional<Vector<Scalar, Dynamic>> SolverLinear(const BlockMatrix& H,
                                                      const Vector<Scalar, Dynamic>& rhs) const {
    if (!computed_ || rhs.size() != dims || !rhs.allFinite()) return std::nullopt;

    if (num_reduced_ == 0) {
      const Vector<Scalar, Dynamic> solution = eliminated_solver.SolverLinear(rhs);
      if (eliminated_solver.Info() != Eigen::Success || !solution.allFinite()) {
        return std::nullopt;
      }
      return solution;
    }

    Vector<Scalar, Dynamic> reduced_rhs = rhs.head(num_reduced_);
    auto accumulate_reduced_rhs = [&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t reduced_type = types.first;
      constexpr size_t eliminated_type = types.second;
      if constexpr (reduced_type < eliminated_type) {
        constexpr int reduced_size = BlockMatrix::kBlockSizes[reduced_type];
        constexpr int eliminated_size = BlockMatrix::kBlockSizes[eliminated_type];
        using EliminatedBlock = Matrix<Scalar, eliminated_size, eliminated_size>;
        using EliminatedVector = Vector<Scalar, eliminated_size>;
        const auto& blocks = H.template blocks<reduced_size, eliminated_size>();
        const auto& lists = coupling_blocks_[K];
        const Index reduced_offset = H.base_offset(reduced_type);
        const Index eliminated_offset = H.base_offset(eliminated_type);
        const auto& inv_vec = std::get<eliminated_type>(eliminated_inverses_);
        for (size_t eliminated_block = 0; eliminated_block < lists.size(); ++eliminated_block) {
          const EliminatedBlock& inverse = inv_vec[eliminated_block];
          const EliminatedVector eliminated_rhs = rhs.template segment<eliminated_size>(
              eliminated_offset + static_cast<Index>(eliminated_block) * eliminated_size);
          const EliminatedVector solved_rhs = inverse * eliminated_rhs;
          for (const CouplingBlock& entry : lists[eliminated_block]) {
            const Index row = reduced_offset + static_cast<Index>(entry.row_block) * reduced_size;
            reduced_rhs.template segment<reduced_size>(row).noalias() -=
                blocks[entry.matrix_block] * solved_rhs;
          }
        }
      }
    };
    VisitBlockPairs(accumulate_reduced_rhs);

    const auto reduced_solution = reduced_solver.SolverLinear(reduced_rhs);
    if (!reduced_solution) return std::nullopt;

    Vector<Scalar, Dynamic> solution(dims);
    solution.head(num_reduced_) = *reduced_solution;
    Vector<Scalar, Dynamic> eliminated_rhs = rhs.tail(num_eliminated_);
    auto back_substitute_coupling = [&](auto pair_index) {
      constexpr size_t K = decltype(pair_index)::value;
      constexpr auto types = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t reduced_type = types.first;
      constexpr size_t eliminated_type = types.second;
      if constexpr (reduced_type < eliminated_type) {
        constexpr int reduced_size = BlockMatrix::kBlockSizes[reduced_type];
        constexpr int eliminated_size = BlockMatrix::kBlockSizes[eliminated_type];
        const auto& blocks = H.template blocks<reduced_size, eliminated_size>();
        const auto& lists = coupling_blocks_[K];
        const Index reduced_offset = H.base_offset(reduced_type);
        const Index eliminated_offset = H.base_offset(eliminated_type) - num_reduced_;
        for (size_t eliminated_block = 0; eliminated_block < lists.size(); ++eliminated_block) {
          auto eliminated_block_rhs = eliminated_rhs.template segment<eliminated_size>(
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
        const Index block_offset = H.base_offset(type);
        if (block_offset < num_reduced_) return;
        const auto& inv_vec = std::get<type>(eliminated_inverses_);
        const size_t n_blocks = static_cast<size_t>(H.num_blocks(type));
        const Index offset_base = block_offset - num_reduced_;
        const Index sol_offset_base = block_offset;
        for (size_t block = 0; block < n_blocks; ++block) {
          const Index off = offset_base + static_cast<Index>(block) * block_size;
          const Index sol_off = sol_offset_base + static_cast<Index>(block) * block_size;
          solution.template segment<block_size>(sol_off).noalias() =
              inv_vec[block] * eliminated_rhs.template segment<block_size>(off);
        }
      }
    };
    VisitBlockPairs(solve_eliminated_blocks);

    if (!solution.allFinite()) return std::nullopt;
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
      const auto& blocks = H.template blocks<block_rows, block_cols>();
      for (size_t k = 0; k < blocks.size(); ++k) {
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
    eliminated_matrix.resize(dims, dims);
    eliminated_matrix.setFromTriplets(triplets.begin(), triplets.end());
    eliminated_matrix.makeCompressed();

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
    return structure_valid;
  }

  bool UpdateFullMatrix(const BlockMatrix& H) {
    auto* values = eliminated_matrix.valuePtr();
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

  Index FindValueIndex(Index row, Index col) const {
    const auto* outer = eliminated_matrix.outerIndexPtr();
    const auto* inner = eliminated_matrix.innerIndexPtr();
    const auto* begin = inner + outer[col];
    const auto* end = inner + outer[col + 1];
    const auto* found = std::lower_bound(begin, end, row);
    if (found == end || *found != row) return -1;
    return static_cast<Index>(found - inner);
  }

  template <typename Block>
  bool InvertAndStore(const Block& block, Block& inverse) {
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

  Index num_eliminated_ = 0;
  DenseMatrixType schur_;
  InversesTupleType eliminated_inverses_;
  std::array<std::vector<BlockIndex>, kNumBlockPairs> structure_;
  std::array<std::vector<MatrixValueSlot>, kNumBlockPairs> full_value_slots_;
  std::array<std::vector<std::vector<CouplingBlock>>, kNumBlockPairs> coupling_blocks_;
  bool analyzed_ = false;
  bool computed_ = false;
};

}  // namespace tinyopt
