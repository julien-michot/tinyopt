// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(TINYOPT_ENABLE_SUITESPARSE)

#include <memory>
#include <optional>
#include <type_traits>

#include <cholmod.h>

#include <tinyopt/types.h>

namespace tinyopt::suitesparse_detail {

struct Common {
  cholmod_common value{};
  bool started = cholmod_start(&value) != 0;

  ~Common() {
    if (started) cholmod_finish(&value);
  }
};

struct FactorDeleter {
  cholmod_common *common;

  void operator()(cholmod_factor *factor) const {
    if (factor) cholmod_free_factor(&factor, common);
  }
};

struct DenseDeleter {
  cholmod_common *common;

  void operator()(cholmod_dense *dense) const {
    if (dense) cholmod_free_dense(&dense, common);
  }
};

template <typename Scalar, int RowsAtCompileTime>
std::optional<Vector<Scalar, RowsAtCompileTime>> SolveWithView(
    const SparseMatrix<Scalar> &A, const Vector<Scalar, RowsAtCompileTime> &b) {
  if (A.rows() != A.cols() || A.rows() != b.size() || A.rows() == 0) return std::nullopt;

  static_assert(std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>);
  static_assert(std::is_same_v<typename SparseMatrix<Scalar>::StorageIndex, int>);

  Common context;
  if (!context.started) return std::nullopt;
  context.value.print = 0;

  cholmod_sparse matrix{};
  matrix.nrow = static_cast<size_t>(A.rows());
  matrix.ncol = static_cast<size_t>(A.cols());
  matrix.nzmax = static_cast<size_t>(A.data().allocatedSize());
  matrix.p = const_cast<int *>(A.outerIndexPtr());
  matrix.i = const_cast<int *>(A.innerIndexPtr());
  matrix.x = const_cast<Scalar *>(A.valuePtr());
  matrix.nz = A.isCompressed() ? nullptr : const_cast<int *>(A.innerNonZeroPtr());
  matrix.stype = 1;
  matrix.itype = CHOLMOD_INT;
  matrix.xtype = CHOLMOD_REAL;
  matrix.dtype = std::is_same_v<Scalar, double> ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE;
  matrix.sorted = 1;
  matrix.packed = A.isCompressed();

  std::unique_ptr<cholmod_factor, FactorDeleter> factor(
      cholmod_analyze(&matrix, &context.value), FactorDeleter{&context.value});
  if (!factor || !cholmod_factorize(&matrix, factor.get(), &context.value) ||
      factor->minor != matrix.ncol)
    return std::nullopt;

  cholmod_dense rhs{};
  rhs.nrow = static_cast<size_t>(b.size());
  rhs.ncol = 1;
  rhs.nzmax = static_cast<size_t>(b.size());
  rhs.d = rhs.nrow;
  rhs.x = const_cast<Scalar *>(b.data());
  rhs.xtype = CHOLMOD_REAL;
  rhs.dtype = matrix.dtype;

  std::unique_ptr<cholmod_dense, DenseDeleter> result(
      cholmod_solve(CHOLMOD_A, factor.get(), &rhs, &context.value),
      DenseDeleter{&context.value});
  if (!result) return std::nullopt;

  using Result = Vector<Scalar, RowsAtCompileTime>;
  Result solution(b.size());
  Eigen::Map<const Vector<Scalar>> result_view(static_cast<const Scalar *>(result->x), b.size());
  solution = result_view;
  if (!solution.allFinite()) return std::nullopt;
  return solution;
}

template <typename Scalar, int RowsAtCompileTime>
std::optional<Vector<Scalar, RowsAtCompileTime>> Solve(
    const SparseMatrix<Scalar> &A, const Vector<Scalar, RowsAtCompileTime> &b) {
  return SolveWithView<Scalar, RowsAtCompileTime>(A, b);
}

}  // namespace tinyopt::suitesparse_detail

#endif