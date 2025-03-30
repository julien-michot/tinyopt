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


template <typename T>
using MatrixBase = Eigen::MatrixBase<T>;
template <typename T>
using DenseBase = Eigen::DenseBase<T>;
template <typename T>
using ArrayBase = Eigen::ArrayBase<T>;

// Let's define some common matrix and vectors

using Mat2 = Eigen::Matrix<double, 2, 2>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat4 = Eigen::Matrix<double, 4, 4>;
using Mat5 = Eigen::Matrix<double, 5, 5>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

using Mat2f = Eigen::Matrix<float, 2, 2>;
using Mat3f = Eigen::Matrix<float, 3, 3>;
using Mat4f = Eigen::Matrix<float, 4, 4>;
using Mat5f = Eigen::Matrix<float, 5, 5>;
using Mat6f = Eigen::Matrix<float, 6, 6>;

using VecX = Eigen::Vector<double, Dynamic>;
using Vec1 = Eigen::Vector<double, 1>;
using Vec2 = Eigen::Vector<double, 2>;
using Vec3 = Eigen::Vector<double, 3>;
using Vec4 = Eigen::Vector<double, 4>;
using Vec5 = Eigen::Vector<double, 5>;
using Vec6 = Eigen::Vector<double, 6>;

using VecXf = Eigen::Vector<float, Dynamic>;
using Vec1f = Eigen::Vector<float, 1>;
using Vec2f = Eigen::Vector<float, 2>;
using Vec3f = Eigen::Vector<float, 3>;
using Vec4f = Eigen::Vector<float, 4>;
using Vec5f = Eigen::Vector<float, 5>;
using Vec6f = Eigen::Vector<float, 6>;

using Mat2X = Eigen::Matrix<double, 2, Eigen::Dynamic>;
using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;

using Mat2Xf = Eigen::Matrix<float, 2, Eigen::Dynamic>;
using Mat3Xf = Eigen::Matrix<float, 3, Eigen::Dynamic>;

using Mat23 = Eigen::Matrix<double, 2, 3>;
using Mat32 = Eigen::Matrix<double, 3, 2>;

using Mat23f = Eigen::Matrix<float, 2, 3>;
using Mat32f = Eigen::Matrix<float, 3, 2>;

/**
 * @brief Computes the inverse of a symmetric, semi-definite matrix.
 *
 * This function calculates the inverse of a symmetric matrix, commonly used for covariance or
 * information matrices. It leverages the LDLT decomposition for efficiency and numerical stability.
 *
 * @tparam Derived The type of the input matrix, which must be a Eigen::MatrixBase.
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
 * @tparam RowsAtCompileTime The number of rows of the matrix, known at compile time.
 * @tparam ColsAtCompileTime The number of columns of the matrix, known at compile time.
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
InvCov(const Derived &m) {
  using MatType =
      Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
  const auto chol = Eigen::SelfAdjointView<const Derived, Eigen::Upper>(m).ldlt();
  if (chol.info() == Eigen::Success && chol.isPositive())
    return chol.solve(MatType::Identity(m.rows(), m.cols()));
  else
    return std::nullopt;
}

/**
 * @brief Solves the linear system A * X = B for X, where A is a symmetric positive-definite matrix, return optional if not.
 *
 * This function utilizes the LDLT decomposition (Cholesky decomposition) to efficiently solve the linear system.
 * It assumes that A is a symmetric positive-definite matrix, which is a requirement for the LDLT decomposition.
 *
 * @tparam Derived The type of the matrix A, which must be a square, symmetric, and positive-definite matrix.
 * @tparam Derived2 The type of the vector B.
 *
 * @param A The coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X if the system is solvable (A is positive-definite),
 * or `std::nullopt` if the system is not solvable (A is not positive-definite).
 *
 * @note The function returns `-chol.solve(b)`, which implies that the solution returned is actually the solution
 * of A * X = -B. If the original equation A * X = B is required, the user should negate the returned vector.
 *
 * @throws (Implicitly) Throws exceptions from Eigen if the LDLT decomposition fails due to reasons other than
 * non-positive definiteness (e.g., memory allocation failure).
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
 *  std::cout << "Expected -B: " << (-b).transpose() << std::endl;
 * } else {
 *  std::cout << "Matrix A is not positive-definite. System cannot be solved." << std::endl;
 * }
 * @endcode
 */
template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>>
Solve(const Derived &A, const Derived2 &b) {
  const auto chol = Eigen::SelfAdjointView<const Derived, Eigen::Upper>(A).ldlt();
  if (chol.info() == Eigen::Success && chol.isPositive()) {
      return chol.solve(b);
  }
  return std::nullopt;
}
}  // namespace tinyopt