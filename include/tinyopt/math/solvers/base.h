// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tinyopt/types.h>
#include <Eigen/SparseCore>

namespace tinyopt {

enum class LinearSolverMethod : uint8_t {
  LDLT,
  LLT,
  LU,
  QR,
  SVD,
  SuiteSparse,  // Cholmod from SuiteSparse
  IncompleteCholesky,
};

constexpr bool RequiresFullMatrix(LinearSolverMethod method) {
  return method != LinearSolverMethod::LDLT && method != LinearSolverMethod::LLT &&
         method != LinearSolverMethod::SuiteSparse &&
         method != LinearSolverMethod::IncompleteCholesky;
}

template <typename Derived>
void CompleteSymmetricMatrix(MatrixBase<Derived> &matrix) {
  matrix.derived() = matrix.derived().template selfadjointView<Eigen::Upper>();
}

template <typename Scalar, int Options, typename StorageIndex>
void CompleteSymmetricMatrix(SparseMatrix<Scalar, Options, StorageIndex> &matrix) {
  SparseMatrix<Scalar, Options, StorageIndex> full =
      matrix.template selfadjointView<Eigen::Upper>();
  matrix.swap(full);
}

struct LinearSolverOptions {
  LinearSolverMethod method = LinearSolverMethod::LDLT;
  bool allow_schur = true;
  /// Relative singular-value cutoff for SVD; 0 uses Eigen's default threshold.
  double svd_relative_threshold = 0.0;
  /// Useful for iterative solvers that update H in place
  bool reuse_symbolic = true;
  /// Initial shift parameter for IncompleteCholesky factorization
  double initial_shift = 1e-3;
};

}  // namespace tinyopt
