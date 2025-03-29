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

template <typename Scalar, int Rows = Dynamic, int Cols = Dynamic, int Options = 0,
          int MaxRows = Rows, int MaxCols = Cols>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

template <typename Scalar, int Rows = Dynamic>
using Vector = Matrix<Scalar, Rows, 1>;

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
  if (chol.isPositive())
    return chol.solve(MatType::Identity(m.rows(), m.cols()));
  else
    return std::nullopt;
}

}  // namespace tinyopt