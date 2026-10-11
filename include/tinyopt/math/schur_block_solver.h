// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/math/linear_solvers.h>
#include <tinyopt/types.h>

namespace tinyopt {

/**
 * @brief Solve H * x = rhs using a TopLeft Schur complement on a block-sparse matrix.
 *
 * The reduced variables occupy the leading `num_reduced` scalar rows and columns. The
 * eliminated corner may contain off-diagonal block couplings. `num_reduced` must match
 * a boundary between the matrix's block-size groups.
 *
 * @return The solution, or std::nullopt if the dimensions or system are invalid or singular.
 */
template <typename Scalar, int... BlockSizes>
std::optional<Vector<Scalar, Dynamic>> SolveBlockSparseSchurTopLeft(
    const BlockSparseMatrix<Scalar, BlockSizes...>& H, const Vector<Scalar, Dynamic>& rhs,
    Index num_reduced, const LinearSolverOptions& options = {}) {
  using BlockMatrix = BlockSparseMatrix<Scalar, BlockSizes...>;
  constexpr size_t kNumBlockTypes = BlockMatrix::kNumBlockTypes;
  constexpr size_t kNumBlockPairs = BlockMatrix::kNumPairs;

  const Index dimension = H.rows();
  if (H.cols() != dimension || rhs.size() != dimension || !rhs.allFinite() || num_reduced <= 0 ||
      num_reduced >= dimension) {
    return std::nullopt;
  }

  bool aligned = false;
  for (size_t type = 0; type < kNumBlockTypes; ++type) {
    if (H.base_offset(type) == num_reduced) {
      aligned = true;
      break;
    }
  }
  if (!aligned) return std::nullopt;

  const Index num_eliminated = dimension - num_reduced;
  Matrix<Scalar, Dynamic, Dynamic> schur =
      Matrix<Scalar, Dynamic, Dynamic>::Zero(num_reduced, num_reduced);
  Matrix<Scalar, Dynamic, Dynamic> coupling =
      Matrix<Scalar, Dynamic, Dynamic>::Zero(num_reduced, num_eliminated);
  std::vector<Eigen::Triplet<Scalar>> eliminated_triplets;
  eliminated_triplets.reserve(static_cast<size_t>(H.nonZeros()));
  bool finite_blocks = true;

  auto visit_blocks = [&]<size_t K>(std::integral_constant<size_t, K>) {
    constexpr auto type_pair = internal::KToPair<kNumBlockTypes>(K);
    constexpr size_t row_type = type_pair.first;
    constexpr size_t col_type = type_pair.second;
    constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
    constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];

    const Index row_offset = H.base_offset(row_type);
    const Index col_offset = H.base_offset(col_type);
    const bool row_reduced = row_offset < num_reduced;
    const bool col_reduced = col_offset < num_reduced;
    const auto& block_indices = H.template block_indices<block_rows, block_cols>();
    const auto& blocks = H.template blocks<block_rows, block_cols>();

    for (size_t k = 0; k < blocks.size(); ++k) {
      const auto& block = blocks[k];
      if (!block.allFinite()) {
        finite_blocks = false;
        continue;
      }

      const Index row = row_offset + block_indices[k].row * block_rows;
      const Index col = col_offset + block_indices[k].col * block_cols;
      const bool has_symmetric_mate =
          row_type != col_type || block_indices[k].row != block_indices[k].col;

      if (row_reduced && col_reduced) {
        schur.block(row, col, block_rows, block_cols) += block;
        if (has_symmetric_mate) {
          schur.block(col, row, block_cols, block_rows) += block.transpose();
        }
      } else if (row_reduced) {
        coupling.block(row, col - num_reduced, block_rows, block_cols) = block;
      } else {
        for (int c = 0; c < block_cols; ++c) {
          for (int r = 0; r < block_rows; ++r) {
            const Index global_row = row - num_reduced + r;
            const Index global_col = col - num_reduced + c;
            const Scalar value = block(r, c);
            if (value != Scalar(0)) {
              eliminated_triplets.emplace_back(global_row, global_col, value);
              if (has_symmetric_mate) {
                eliminated_triplets.emplace_back(global_col, global_row, value);
              }
            }
          }
        }
      }
    }
  };

  [&]<size_t... Ks>(std::index_sequence<Ks...>) {
    (visit_blocks(std::integral_constant<size_t, Ks>{}), ...);
  }(std::make_index_sequence<kNumBlockPairs>{});
  if (!finite_blocks) return std::nullopt;

  SparseMatrix<Scalar> eliminated(num_eliminated, num_eliminated);
  eliminated.setFromTriplets(eliminated_triplets.begin(), eliminated_triplets.end());
  SparseLinearSolvers<SparseMatrix<Scalar>> eliminated_solver(options.method, num_eliminated,
                                                              options);
  if (!eliminated_solver.compute(eliminated) || eliminated_solver.info() != Eigen::Success) {
    return std::nullopt;
  }

  const auto rhs_eliminated = rhs.tail(num_eliminated);
  const Vector<Scalar, Dynamic> solved_rhs = eliminated_solver.solve(rhs_eliminated);
  if (eliminated_solver.info() != Eigen::Success || !solved_rhs.allFinite()) {
    return std::nullopt;
  }
  Vector<Scalar, Dynamic> reduced_rhs = rhs.head(num_reduced);
  reduced_rhs.noalias() -= coupling * solved_rhs;

  const Matrix<Scalar, Dynamic, Dynamic> solved_coupling =
      eliminated_solver.solve(coupling.transpose());
  if (eliminated_solver.info() != Eigen::Success || !solved_coupling.allFinite()) {
    return std::nullopt;
  }

  auto accumulate_schur_update = [&]<size_t K>(std::integral_constant<size_t, K>) {
    constexpr auto type_pair = internal::KToPair<kNumBlockTypes>(K);
    constexpr size_t row_type = type_pair.first;
    constexpr size_t col_type = type_pair.second;
    constexpr int block_rows = BlockMatrix::kBlockSizes[row_type];
    constexpr int block_cols = BlockMatrix::kBlockSizes[col_type];

    if constexpr (row_type < col_type) {
      const Index row_offset = H.base_offset(row_type);
      const Index col_offset = H.base_offset(col_type);
      if (row_offset >= num_reduced || col_offset < num_reduced) return;

      const auto& block_indices = H.template block_indices<block_rows, block_cols>();
      const auto& blocks = H.template blocks<block_rows, block_cols>();
      for (size_t k = 0; k < blocks.size(); ++k) {
        const Index row = row_offset + block_indices[k].row * block_rows;
        const Index col = col_offset + block_indices[k].col * block_cols - num_reduced;
        schur.block(row, 0, block_rows, num_reduced).noalias() -=
            blocks[k] * solved_coupling.block(col, 0, block_cols, num_reduced);
      }
    }
  };

  [&]<size_t... Ks>(std::index_sequence<Ks...>) {
    (accumulate_schur_update(std::integral_constant<size_t, Ks>{}), ...);
  }(std::make_index_sequence<kNumBlockPairs>{});

  CompleteSymmetricMatrix(schur);
  if (!schur.allFinite() || !reduced_rhs.allFinite()) return std::nullopt;

  LinearSolverOptions dense_options = options;
  if (dense_options.method == LinearSolverMethod::SuiteSparse ||
      dense_options.method == LinearSolverMethod::IncompleteCholesky) {
    dense_options.method = LinearSolverMethod::LDLT;
  }
  DenseLinearSolvers<Matrix<Scalar, Dynamic, Dynamic>> reduced_solver(dense_options, num_reduced);
  const auto reduced_solution = reduced_solver.solve(schur, reduced_rhs);
  if (!reduced_solution) return std::nullopt;

  Vector<Scalar, Dynamic> solution(dimension);
  solution.head(num_reduced) = *reduced_solution;
  solution.tail(num_eliminated) =
      eliminated_solver.solve(rhs_eliminated - coupling.transpose() * solution.head(num_reduced));
  if (eliminated_solver.info() != Eigen::Success || !solution.allFinite()) {
    return std::nullopt;
  }
  return solution;
}

}  // namespace tinyopt
