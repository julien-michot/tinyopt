// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <type_traits>

#include <Eigen/SparseCholesky>
#include <tinyopt/types.h>

namespace tinyopt {

template <typename Derived>
std::optional<Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime,
                     Derived::ColsAtCompileTime>>
DenseInvCov(const MatrixBase<Derived> &m) {
  using MatType =
      Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
  if constexpr (Derived::ColsAtCompileTime == 1) {
    return MatType(m.cwiseInverse().asDiagonal());
  } else if (m.cols() == 1) {
    return MatType(m.inverse());
  } else if (m.cols() > 0) {
    const auto &chol = m.template selfadjointView<Upper>().ldlt();
    if (chol.info() == Eigen::Success && chol.isPositive())
      return chol.solve(MatType::Identity(m.rows(), m.cols()));
  }
  return std::nullopt;
}

template <typename Derived>
auto InvCov(const MatrixBase<Derived> &m) {
  return DenseInvCov(m);
}

template <typename T>
std::optional<SparseMatrix<typename T::Scalar>> SparseInvCov(
    const T &m, typename T::Scalar retry_with_shift_offset = typename T::Scalar(0.0)) {
  using Scalar = typename T::Scalar;
  if (m.size() == 0) return std::nullopt;
  Eigen::SimplicialLDLT<SparseMatrix<Scalar>, Eigen::Upper> solver;
  solver.compute(m);
  if (solver.info() != Eigen::Success) {
    if (retry_with_shift_offset > 0 && solver.info() == Eigen::NumericalIssue) {
      solver.setShift(retry_with_shift_offset);
      solver.compute(m);
    }
    if (solver.info() != Eigen::Success) return std::nullopt;
  }
  SparseMatrix<Scalar> identity(m.rows(), m.cols());
  identity.setIdentity();
  auto inverse = solver.solve(identity);
  if (solver.info() != Eigen::Success) return std::nullopt;
  return inverse;
}

template <typename Scalar>
std::optional<SparseMatrix<Scalar>> InvCov(
    const SparseMatrix<Scalar> &m, Scalar retry_with_shift_offset = Scalar(0.0)) {
  return SparseInvCov(m, retry_with_shift_offset);
}

template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
auto InvCov(const Block<XprType, BlockRows, BlockCols, InnerPanel> &m) {
  using Scalar = typename XprType::Scalar;
  if constexpr (std::is_same_v<std::decay_t<XprType>, SparseMatrix<Scalar>>)
    return SparseInvCov(m);
  else
    return DenseInvCov(m);
}

}  // namespace tinyopt