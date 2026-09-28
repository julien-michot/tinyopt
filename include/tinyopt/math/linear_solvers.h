// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <Eigen/SparseCholesky>

#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
#include <Eigen/Cholesky>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
#include <Eigen/LU>
#include <Eigen/SparseLU>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
#include <Eigen/QR>
#include <Eigen/SparseQR>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
#include <Eigen/SVD>
#endif

#include <tinyopt/types.h>

namespace tinyopt {

enum class LinearSolverMethod : uint8_t {
  LDLT,
  LLT,
  LU,
  QR,
  SVD,
};

constexpr bool RequiresFullMatrix(LinearSolverMethod method) {
  return method != LinearSolverMethod::LDLT && method != LinearSolverMethod::LLT;
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

template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>> SolveLDLT(
    const MatrixBase<Derived> &A, const MatrixBase<Derived2> &b) {
  using Scalar = typename Derived::Scalar;
  using Result = Vector<Scalar, Derived::RowsAtCompileTime>;
  const auto decomposition = A.template selfadjointView<Eigen::Upper>().ldlt();
  if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) return std::nullopt;
  Result solution = decomposition.solve(b);
  if (decomposition.info() != Eigen::Success) return std::nullopt;
  // NOTE: solution.allFinite() is tested outside of this function in SolveLinearSystem() to avoid double-checking.
  return solution;
}

template <typename Scalar, int RowsAtCompileTime = Dynamic>
std::optional<Vector<Scalar, RowsAtCompileTime>> SolveLDLT(
    const SparseMatrix<Scalar> &A, const Vector<Scalar, RowsAtCompileTime> &b) {
  Eigen::SimplicialLDLT<SparseMatrix<Scalar>, Eigen::Upper> decomposition;
  decomposition.compute(A);
  if (decomposition.info() != Eigen::Success) return std::nullopt;
  Vector<Scalar, RowsAtCompileTime> solution = decomposition.solve(b);
  if (decomposition.info() != Eigen::Success) return std::nullopt;
  // NOTE: solution.allFinite() is tested outside of this function in SolveLinearSystem() to avoid double-checking.
  return solution;
}

template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>> SolveLinearSystem(
    const MatrixBase<Derived> &A, const MatrixBase<Derived2> &b, LinearSolverMethod method) {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT) || defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU) || \
    defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR) || defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
  using Scalar = typename Derived::Scalar;
  using MatrixType = typename Derived::PlainObject;
  using Result = Vector<Scalar, Derived::RowsAtCompileTime>;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
  const Scalar epsilon = Eigen::NumTraits<Scalar>::epsilon();
#endif

  switch (method) {
    case LinearSolverMethod::LDLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
      return SolveLDLT(A, b);
#else
      return std::nullopt;
#endif
    case LinearSolverMethod::LLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
      Eigen::LLT<MatrixType, Eigen::Upper> decomposition(A);
      if (decomposition.info() != Eigen::Success) return std::nullopt;
      Result solution = decomposition.solve(b);
      if (decomposition.info() != Eigen::Success || !solution.allFinite()) return std::nullopt;
      return solution;
#else
      return std::nullopt;
#endif
    }
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
    case LinearSolverMethod::LU: {
      Eigen::PartialPivLU<MatrixType> decomposition(A);
      if (decomposition.rcond() <= epsilon) return std::nullopt;
      Result solution = decomposition.solve(b);
      if (!solution.allFinite()) return std::nullopt;
      return solution;
    }
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
    case LinearSolverMethod::QR: {
      Eigen::ColPivHouseholderQR<MatrixType> decomposition(A);
      if (decomposition.rank() < A.cols()) return std::nullopt;
      Result solution = decomposition.solve(b);
      if (!solution.allFinite()) return std::nullopt;
      return solution;
    }
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
    case LinearSolverMethod::SVD: {
      Eigen::JacobiSVD<MatrixType> decomposition(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      if (decomposition.rank() < A.cols()) return std::nullopt;
      Result solution = decomposition.solve(b);
      if (!solution.allFinite()) return std::nullopt;
      return solution;
    }
#endif
    default:
      return std::nullopt;
  }
  return std::nullopt;
}

template <typename Scalar, int RowsAtCompileTime = Dynamic>
std::optional<Vector<Scalar, RowsAtCompileTime>> SolveLinearSystem(
    const SparseMatrix<Scalar> &A, const Vector<Scalar, RowsAtCompileTime> &b,
    LinearSolverMethod method) {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT) || defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU) || \
    defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
  using Result = Vector<Scalar, RowsAtCompileTime>;
#endif
  switch (method) {
    case LinearSolverMethod::LDLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
      return SolveLDLT(A, b);
#else
      return std::nullopt;
#endif
    case LinearSolverMethod::LLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
      Eigen::SimplicialLLT<SparseMatrix<Scalar>, Eigen::Upper> decomposition;
      decomposition.compute(A);
      if (decomposition.info() != Eigen::Success) return std::nullopt;
      Result solution = decomposition.solve(b);
      if (decomposition.info() != Eigen::Success || !solution.allFinite()) return std::nullopt;
      return solution;
#else
      return std::nullopt;
#endif
    }
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
    case LinearSolverMethod::LU: {
      Eigen::SparseLU<SparseMatrix<Scalar>> decomposition;
      decomposition.compute(A);
      if (decomposition.info() != Eigen::Success) return std::nullopt;
      Result solution = decomposition.solve(b);
      if (decomposition.info() != Eigen::Success || !solution.allFinite()) return std::nullopt;
      return solution;
    }
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
    case LinearSolverMethod::QR: {
      Eigen::SparseQR<SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>> decomposition;
      decomposition.compute(A);
      if (decomposition.info() != Eigen::Success || decomposition.rank() < A.cols())
        return std::nullopt;
      Result solution = decomposition.solve(b);
      if (decomposition.info() != Eigen::Success || !solution.allFinite()) return std::nullopt;
      return solution;
    }
#endif
    case LinearSolverMethod::SVD:
      return std::nullopt;
    default:
      return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace tinyopt