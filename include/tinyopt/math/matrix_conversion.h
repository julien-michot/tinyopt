// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/traits/matrix_traits.h>
#include <tinyopt/types.h>

namespace tinyopt::internal {

template <typename M>
bool ChooseBlockCounts(Index dims, std::array<Index, M::kNumBlockTypes>& counts) {
  std::vector<Index> previous_dims(static_cast<size_t>(dims + 1), Index(-1));
  std::vector<size_t> previous_types(static_cast<size_t>(dims + 1), M::kNumBlockTypes);
  previous_dims[0] = 0;
  for (Index current = 0; current <= dims; ++current) {
    if (previous_dims[static_cast<size_t>(current)] == -1) continue;
    for (size_t type = 0; type < M::kNumBlockTypes; ++type) {
      const Index next = current + M::kBlockSizes[type];
      if (next <= dims && previous_dims[static_cast<size_t>(next)] == -1) {
        previous_dims[static_cast<size_t>(next)] = current;
        previous_types[static_cast<size_t>(next)] = type;
      }
    }
  }
  if (previous_dims[static_cast<size_t>(dims)] == -1) return false;
  for (Index current = dims; current > 0;) {
    const size_t type = previous_types[static_cast<size_t>(current)];
    ++counts[type];
    current = previous_dims[static_cast<size_t>(current)];
  }
  return true;
}

template <typename M, size_t Row, size_t Col = Row>
void FillBlockSparseCols(M& matrix, const MatX& dense,
                         const std::array<Index, M::kNumBlockTypes>& counts,
                         const std::array<Index, M::kNumBlockTypes>& offsets) {
  if constexpr (Col < M::kNumBlockTypes) {
    if constexpr (Col >= Row) {
      constexpr int BlockRows = M::kBlockSizes[Row];
      constexpr int BlockCols = M::kBlockSizes[Col];
      for (Index row = 0; row < counts[Row]; ++row) {
        for (Index col = 0; col < counts[Col]; ++col) {
          matrix.template block_by_type<Row, Col>(row, col) = dense.block(
              offsets[Row] + row * BlockRows, offsets[Col] + col * BlockCols, BlockRows, BlockCols);
        }
      }
    }
    FillBlockSparseCols<M, Row, Col + 1>(matrix, dense, counts, offsets);
  }
}

template <typename M, size_t Row>
void FillBlockSparseRows(M& matrix, const MatX& dense,
                         const std::array<Index, M::kNumBlockTypes>& counts,
                         const std::array<Index, M::kNumBlockTypes>& offsets) {
  if constexpr (Row < M::kNumBlockTypes) {
    FillBlockSparseCols<M, Row>(matrix, dense, counts, offsets);
    FillBlockSparseRows<M, Row + 1>(matrix, dense, counts, offsets);
  }
}

template <typename M>
void FillBlockSparse(M& matrix, const MatX& dense,
                     const std::array<Index, M::kNumBlockTypes>& counts) {
  std::array<Index, M::kNumBlockTypes> offsets{};
  for (size_t i = 1; i < M::kNumBlockTypes; ++i) {
    offsets[i] = offsets[i - 1] + counts[i - 1] * M::kBlockSizes[i - 1];
  }
  FillBlockSparseRows<M, 0>(matrix, dense, counts, offsets);
}

template <typename M, typename InverseType, typename Scale>
std::optional<M> ConvertMatrix(const InverseType& inverse, Scale scale,
                               const std::vector<int>& source_sizes = {},
                               const std::vector<Index>& source_counts = {}) {
  if constexpr (traits::is_block_tridiag_matrix_v<M>) {
    using TargetScalar = typename M::Scalar;
    MatX dense = (MatX(inverse) * scale).template cast<double>();
    const int B =
        (M::kBlockSize != Dynamic) ? M::kBlockSize : (source_sizes.empty() ? 1 : source_sizes[0]);
    const Index N = dense.rows() / B;
    M covariance(N, B);
    for (Index i = 0; i < N; ++i) {
      covariance.diagonal(i) = dense.block(i * B, i * B, B, B).template cast<TargetScalar>();
    }
    for (Index i = 0; i < N - 1; ++i) {
      covariance.off_diagonal(i) =
          dense.block(i * B, (i + 1) * B, B, B).template cast<TargetScalar>();
    }
    return covariance;
  } else if constexpr (requires { M::kNumBlockTypes; }) {
    MatX dense = MatX(inverse) * scale;
    constexpr size_t NumTypes = M::kNumBlockTypes;
    std::array<Index, NumTypes> counts{};
    bool matching_layout = source_sizes.size() == NumTypes && source_counts.size() == NumTypes;
    for (size_t i = 0; i < NumTypes && matching_layout; ++i) {
      matching_layout = source_sizes[i] == M::kBlockSizes[i];
      counts[i] = source_counts[i];
    }
    if (!matching_layout && !ChooseBlockCounts<M>(dense.rows(), counts)) return {};

    Index expected_rows = 0;
    for (size_t i = 0; i < NumTypes; ++i) {
      expected_rows += counts[i] * M::kBlockSizes[i];
    }
    if (expected_rows != dense.rows()) return {};

    M covariance(counts);
    FillBlockSparse(covariance, dense, counts);
    return covariance;
  } else if constexpr (traits::is_block_sparse_matrix_v<M>) {
    using TargetScalar = typename M::Scalar;
    SparseMatrix<TargetScalar> sparse_inverse;
    if constexpr (traits::is_sparse_matrix_v<InverseType>) {
      sparse_inverse = inverse.template cast<TargetScalar>() * scale;
    } else {
      sparse_inverse = (inverse.template cast<TargetScalar>() * scale).sparseView();
    }
    if (source_sizes.empty() || source_counts.empty()) {
      return M(std::move(sparse_inverse), {1}, {inverse.rows()});
    }
    return M(std::move(sparse_inverse), source_sizes, source_counts);
  } else if constexpr (traits::is_sparse_matrix_v<M>) {
    using TargetScalar = typename M::Scalar;
    if (M::RowsAtCompileTime != Dynamic && M::RowsAtCompileTime != inverse.rows()) return {};
    if (M::ColsAtCompileTime != Dynamic && M::ColsAtCompileTime != inverse.cols()) return {};
    if constexpr (traits::is_sparse_matrix_v<InverseType>) {
      return M(inverse.template cast<TargetScalar>() * scale);
    } else {
      auto scaled = inverse.template cast<TargetScalar>() * scale;
      return M(scaled.sparseView());
    }
  } else {
    using TargetScalar = typename M::Scalar;
    if (M::RowsAtCompileTime != Dynamic && M::RowsAtCompileTime != inverse.rows()) return {};
    if (M::ColsAtCompileTime != Dynamic && M::ColsAtCompileTime != inverse.cols()) return {};
    return M(inverse.template cast<TargetScalar>() * scale);
  }
}

}  // namespace tinyopt::internal
