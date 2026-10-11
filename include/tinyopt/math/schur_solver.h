// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <Eigen/SparseCholesky>

#include <tinyopt/math/linear_solvers.h>
#include <tinyopt/types.h>

namespace tinyopt {

template <typename Scalar, int Options = 0, typename StorageIndex = int,
          typename Derived = Vector<Scalar, Dynamic>>
std::optional<Vector<Scalar, Dynamic>> SolveSparseSchurTopLeft(
    const SparseMatrix<Scalar, Options, StorageIndex> &H, const MatrixBase<Derived> &rhs,
    Index num_reduced,
    SparseLinearSolvers<SparseMatrix<Scalar, Options, StorageIndex>> &solver,
    const LinearSolverOptions &options = {}) {
  const Index D = H.rows();
  const Index N = num_reduced;
  if (N <= 0 || N >= D || H.cols() != D || rhs.size() != D) {
    return std::nullopt;
  }
  const Index M = D - N;

  // 1. Factorize eliminated block C (bottom-right M x M)
  if (!solver.compute(H.bottomRightCorner(M, M)) || solver.info() != Eigen::Success) {
    return std::nullopt;
  }

  // 2. Use Eigen sparse block expressions; ET is converted to dense for solver
  const auto E = H.topRightCorner(N, M);
  const auto ET = Matrix<Scalar, Dynamic, Dynamic>(E.transpose());
  auto Y = solver.solve(ET);

  // 3. Form reduced Schur complement system S = B - E * Y
  Matrix<Scalar, Dynamic, Dynamic> S = H.topLeftCorner(N, N);
  CompleteSymmetricMatrix(S);
  S.noalias() -= E * Y;
  if (!S.allFinite()) {
    return std::nullopt;
  }

  // 4. Compute reduced rhs expression g_S = b_r - E * C^{-1} * b_e using Eigen glue classes
  Vector<Scalar, Dynamic> g_S = rhs.head(N) - E * solver.solve(rhs.tail(M));

  // 5. Solve S * dx_r = g_S
  LinearSolverOptions dense_opt = options;
  if (dense_opt.method == LinearSolverMethod::SuiteSparse ||
      dense_opt.method == LinearSolverMethod::IncompleteCholesky) {
    dense_opt.method = LinearSolverMethod::LDLT;
  }
  DenseLinearSolvers<Matrix<Scalar, Dynamic, Dynamic>> solver_S(dense_opt, N);
  auto maybe_dx_r = solver_S.solve(S, g_S);
  if (!maybe_dx_r) {
    return std::nullopt;
  }

  // 6. Back-substitute dx_e = C^{-1} * (b_e - E^T * dx_r) directly into dx.tail(M)
  Vector<Scalar, Dynamic> dx(D);
  dx.head(N) = *maybe_dx_r;
  dx.tail(M) = solver.solve(rhs.tail(M) - E.transpose() * dx.head(N));
  if (solver.info() != Eigen::Success || !dx.allFinite()) {
    return std::nullopt;
  }

  return dx;
}

template <typename Scalar, int Options = 0, typename StorageIndex = int,
          typename Derived = Vector<Scalar, Dynamic>>
std::optional<Vector<Scalar, Dynamic>> SolveSparseSchurTopLeft(
    const SparseMatrix<Scalar, Options, StorageIndex> &H, const MatrixBase<Derived> &rhs,
    Index num_reduced, const LinearSolverOptions &options = {}) {
  SparseLinearSolvers<SparseMatrix<Scalar, Options, StorageIndex>> solver(
      options.method, H.rows() - num_reduced, options);
  return SolveSparseSchurTopLeft(H, rhs, num_reduced, solver, options);
}

}  // namespace tinyopt
