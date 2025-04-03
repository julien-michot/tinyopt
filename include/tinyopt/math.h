// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Eigen/src/SparseCore/SparseSelfAdjointView.h>
#include <Eigen/SparseCholesky>

#include <optional>

namespace tinyopt {

static constexpr int Dynamic = Eigen::Dynamic;
static constexpr int Lower = Eigen::Lower;
static constexpr int Upper = Eigen::Upper;

template <typename Scalar, int Rows = Dynamic, int Cols = Dynamic, int Options = 0,
          int MaxRows = Rows, int MaxCols = Cols>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

template <typename Scalar, int Rows = Dynamic>
using Vector = Matrix<Scalar, Rows, 1>;

template <typename T, int Cols>
using RowVector = Matrix<T, 1, Cols, Eigen::RowMajor>;

template <typename Scalar = double, int Options = 0, typename StorageIndex = int>
using SparseMatrix = Eigen::SparseMatrix<Scalar, Options, StorageIndex>;

using SparseMatrixf = SparseMatrix<float>;

template <typename T>
using MatrixBase = Eigen::MatrixBase<T>;
template <typename T>
using DenseBase = Eigen::DenseBase<T>;
template <typename T>
using ArrayBase = Eigen::ArrayBase<T>;

// Let's define some common matrix and vectors
using MatX = Matrix<double, Dynamic, Dynamic>;
using MatXf = Matrix<float, Dynamic, Dynamic>;

using Mat1 = Matrix<double, 1, 1>;
using Mat2 = Matrix<double, 2, 2>;
using Mat3 = Matrix<double, 3, 3>;
using Mat4 = Matrix<double, 4, 4>;
using Mat5 = Matrix<double, 5, 5>;
using Mat6 = Matrix<double, 6, 6>;

using Mat1f = Matrix<float, 1, 1>;
using Mat2f = Matrix<float, 2, 2>;
using Mat3f = Matrix<float, 3, 3>;
using Mat4f = Matrix<float, 4, 4>;
using Mat5f = Matrix<float, 5, 5>;
using Mat6f = Matrix<float, 6, 6>;

using VecX = Vector<double, Dynamic>;
using Vec1 = Vector<double, 1>;
using Vec2 = Vector<double, 2>;
using Vec3 = Vector<double, 3>;
using Vec4 = Vector<double, 4>;
using Vec5 = Vector<double, 5>;
using Vec6 = Vector<double, 6>;

using VecXf = Vector<float, Dynamic>;
using Vec1f = Vector<float, 1>;
using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;
using Vec5f = Vector<float, 5>;
using Vec6f = Vector<float, 6>;

using Mat2X = Matrix<double, 2, Dynamic>;
using Mat3X = Matrix<double, 3, Dynamic>;

using Mat2Xf = Matrix<float, 2, Dynamic>;
using Mat3Xf = Matrix<float, 3, Dynamic>;

using Mat23 = Matrix<double, 2, 3>;
using Mat32 = Matrix<double, 3, 2>;

using Mat23f = Matrix<float, 2, 3>;
using Mat32f = Matrix<float, 3, 2>;

/**
 * @brief Computes the inverse of a symmetric, semi-definite matrix.
 *
 * This function calculates the inverse of a symmetric matrix, commonly used for covariance or
 * information matrices. It leverages the LDLT decomposition for efficiency and numerical stability.
 *
 * @param[in] m The symmetric input matrix. It can be filled either fully or only in the upper
 * triangular part.
 *
 * @return The inverse of the input matrix or nullopt, with the same dimensions and scalar type as
 * the input.
 *
 * @note The input matrix is assumed to be symmetric. If only the upper triangular part is filled,
 * the function implicitly uses the symmetry to construct the full matrix for the LDLT
 * decomposition.
 *
 * @tparam Scalar The scalar type of the matrix elements.
 *
 * @code
 * Eigen::MatrixXd covarianceMatrix(3, 3);
 * covarianceMatrix << 1.0, 0.5, 0.2,
 * 0.5, 2.0, 0.8,
 * 0.2, 0.8, 3.0;
 *
 * auto inverseCovariance = InvCov(covarianceMatrix);
 * if (inverseCovariance)
 *   std::cout << "Inverse Covariance Matrix:\n" << inverseCovariance.value() << std::endl;
 * @endcode
 */
template <typename Derived>
std::optional<
    Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
InvCov(const MatrixBase<Derived> &m) {
  using MatType =
      Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
  const auto chol = Eigen::SelfAdjointView<const Derived, Eigen::Upper>(m).ldlt();
  if (chol.info() == Eigen::Success && chol.isPositive())
    return chol.solve(MatType::Identity(m.rows(), m.cols()));
  else
    return std::nullopt;
}

/**
 * @brief Computes the inverse of a symmetric, semi-definite sparse matrix.
 *
 * This function calculates the inverse of a symmetric matrix, commonly used for covariance or
 * information matrices. It leverages the LDLT decomposition for efficiency and numerical stability.
 *
 * @param[in] m The symmetric input matrix. It can be filled either fully or only in the upper
 * triangular part.
 *
 * @return The inverse of the input matrix or nullopt, with the same dimensions and scalar type as
 * the input.
 *
 * @note The input matrix is assumed to be symmetric. If only the upper triangular part is filled,
 * the function implicitly uses the symmetry to construct the full matrix for the LDLT
 * decomposition.
 *
 * @tparam Scalar The scalar type of the matrix elements.
 */
template <typename Scalar>
std::optional<SparseMatrix<Scalar>> InvCov(const SparseMatrix<Scalar> &m) {
  Eigen::SimplicialLDLT<SparseMatrix<Scalar>, Eigen::Upper> solver;
  solver.compute(m);
  if (solver.info() != Eigen::Success)  // Decomposition failed
    return std::nullopt;
  SparseMatrix<Scalar> I(m.rows(), m.cols());
  I.setIdentity();
  auto X = solver.solve(I);
  if (solver.info() != Eigen::Success)  // Solving failed
    return std::nullopt;
  return X;
}

/**
 * @brief Solves the linear system A * X = B for X, where A is a symmetric positive-definite matrix,
 * return optional otherwise.
 *
 * This function utilizes the LDLT decomposition (Cholesky decomposition) to efficiently solve the
 * linear system. It assumes that A is a symmetric positive-definite matrix, which is a requirement
 * for the LDLT decomposition.
 *
 * @tparam Derived The type of the matrix A, which must be a square, symmetric, and
 * positive-definite matrix.
 * @tparam Derived2 The type of the vector B.
 *
 * @param A The coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X if the system is solvable (A is
 * positive-definite), or `std::nullopt` if the system is not solvable (A is not positive-definite).
 *
 * @note The function returns `chol.solve(b)`, which implies that the solution returned is actually
 * the solution of A * X = B.
 *
 * @throws (Implicitly) Throws exceptions from Eigen if the LDLT decomposition fails due to reasons
 * other than non-positive definiteness (e.g., memory allocation failure).
 *
 * @code
 * Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);
 * A = A * A.transpose(); // Ensure A is symmetric positive-definite (or semi-definite)
 * Eigen::VectorXd b = Eigen::VectorXd::Random(3);
 *
 * auto x_opt = Solve(A, b);
 *
 * if (x_opt) {
 *  Eigen::VectorXd x = x_opt.value();
 *  std::cout << "Solution X: " << x.transpose() << std::endl;
 *  std::cout << "A * X: " << (A * x).transpose() << std::endl;
 *  std::cout << "Expected -B: " << b.transpose() << std::endl;
 * } else {
 *  std::cout << "Matrix A is not positive-definite. System cannot be solved." << std::endl;
 * }
 * @endcode
 */
template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>> SolveLDLT(
    const MatrixBase<Derived> &A, const MatrixBase<Derived2> &b) {
  const auto chol = Eigen::SelfAdjointView<const Derived, Eigen::Upper>(A).ldlt();
  if (chol.info() == Eigen::Success && chol.isPositive()) {
    return chol.solve(b);
  }
  return std::nullopt;
}

/**
 * @brief Solves the linear system A * X = B for X, where A is a symmetric positive-definite sparse
 * matrix, return optional otherwise.
 *
 * This function utilizes the LDLT decomposition (Cholesky decomposition) to efficiently solve the
 * linear system. It assumes that A is a symmetric positive-definite matrix, which is a requirement
 * for the LDLT decomposition.
 *
 * @tparam Derived The type of the matrix A, which must be a square, symmetric, and
 * positive-definite matrix.
 * @tparam Derived2 The type of the vector B.
 *
 * @param A The sparse coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X if the system is solvable (A is
 * positive-definite), or `std::nullopt` if the system is not solvable (A is not positive-definite).
 *
 * @note The function returns `chol.solve(b)`, which implies that the solution returned is actually
 * the solution of A * X = B.
 *
 * @throws (Implicitly) Throws exceptions from Eigen if the LDLT decomposition fails due to reasons
 * other than non-positive definiteness (e.g., memory allocation failure).
 */
template <typename Scalar, int RowsAtCompileTime>
std::optional<Vector<Scalar, RowsAtCompileTime>> SolveLDLT(
    const SparseMatrix<Scalar> &A, const Vector<Scalar, RowsAtCompileTime> &b) {
  Eigen::SimplicialLDLT<SparseMatrix<Scalar>, Eigen::Upper> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success)  // Decomposition failed
    return std::nullopt;
  auto X = solver.solve(b);
  if (solver.info() != Eigen::Success)  // Solving failed
    return std::nullopt;
  return X;
}

/// Integer square root function
constexpr inline int SQRT(int N) {
  if (N <= 1) return N;
  int left = 1, right = N / 2;
  int result = 0;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (mid <= N / mid) {
      left = mid + 1;
      result = mid;
    } else {
      right = mid - 1;
    }
  }
  return result;
};

}  // namespace tinyopt