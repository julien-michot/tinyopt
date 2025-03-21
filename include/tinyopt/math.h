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

#include <sstream>

namespace tinyopt {

static constexpr int Dynamic = Eigen::Dynamic;

template <typename Scalar, int Rows = Dynamic, int Cols = Dynamic,
          int Options = 0, int MaxRows = Rows, int MaxCols = Cols>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

template <typename Scalar, int Rows = Dynamic>
using Vector = Matrix<Scalar, Rows, 1>;

// Inverse a symmetric matrix (typically a cov or info matrix)
template <typename Derived>
Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
InvCov(const Derived &m) {
  using MatType =
      Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
  const auto chol = Eigen::SelfAdjointView<const Derived, Eigen::Upper>(m).ldlt();
  return chol.solve(MatType::Identity(m.rows(), m.cols()));
}

// Convert a matrix to a string using default formatting
template <typename Derived>
std::string toString(const Eigen::MatrixBase<Derived> &m) {
  std::stringstream ss;
  if (m.cols() == 1)
    ss << m.transpose();
  else
    ss << m;
  return ss.str();
}

inline std::string toString(const double &m) {
  std::stringstream ss;
  ss << m;
  return ss.str();
}

} // namespace tinyopt