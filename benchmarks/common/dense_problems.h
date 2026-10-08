// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Eigen/Core>

namespace tinyopt::benchmark {

template <typename Derived>
auto DenseMathResiduals(const Eigen::MatrixBase<Derived>& x) {
  using Scalar = typename Derived::Scalar;
  Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 1> residuals(x.size());
  if constexpr (Derived::RowsAtCompileTime == 1) {
    residuals[0] = x[0] * x[0] - Scalar(2);
  } else if constexpr (Derived::RowsAtCompileTime == 2) {
    residuals[0] = x[0] * x[0] + x[1] - Scalar(2);
    residuals[1] = x[0] + x[1] * x[1] - Scalar(2);
  } else if constexpr (Derived::RowsAtCompileTime == 3) {
    residuals[0] = x[0] * x[0] + x[1] + x[2] - Scalar(3);
    residuals[1] = x[0] + x[1] * x[1] + x[2] - Scalar(3);
    residuals[2] = x[0] + x[1] + x[2] * x[2] - Scalar(3);
  } else if (x.size() == 1) {
    residuals[0] = x[0] * x[0] - Scalar(2);
  } else if (x.size() == 2) {
    residuals[0] = x[0] * x[0] + x[1] - Scalar(2);
    residuals[1] = x[0] + x[1] * x[1] - Scalar(2);
  } else {
    residuals[0] = x[0] * x[0] + x[1] + x[2] - Scalar(3);
    residuals[1] = x[0] + x[1] * x[1] + x[2] - Scalar(3);
    residuals[2] = x[0] + x[1] + x[2] * x[2] - Scalar(3);
  }
  return residuals;
}

template <typename Parameters, typename Jacobian>
void DenseMathJacobian(const Eigen::MatrixBase<Parameters>& x,
                       Eigen::MatrixBase<Jacobian>& jacobian) {
  using Scalar = typename Parameters::Scalar;
  auto& output = jacobian.derived();
  output.setZero();
  if constexpr (Parameters::RowsAtCompileTime == 1) {
    output(0, 0) = Scalar(2) * x[0];
  } else if constexpr (Parameters::RowsAtCompileTime == 2) {
    output(0, 0) = Scalar(2) * x[0];
    output(0, 1) = Scalar(1);
    output(1, 0) = Scalar(1);
    output(1, 1) = Scalar(2) * x[1];
  } else if constexpr (Parameters::RowsAtCompileTime == 3) {
    output.setOnes();
    output.diagonal() = Scalar(2) * x;
  } else if (x.size() == 1) {
    output(0, 0) = Scalar(2) * x[0];
  } else if (x.size() == 2) {
    output(0, 0) = Scalar(2) * x[0];
    output(0, 1) = Scalar(1);
    output(1, 0) = Scalar(1);
    output(1, 1) = Scalar(2) * x[1];
  } else {
    output.setOnes();
    output.diagonal() = Scalar(2) * x;
  }
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DenseMathInitial(Eigen::Index dimensions) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> initial(dimensions);
  if (dimensions == 1) {
    initial[0] = Scalar(1);
  } else if (dimensions == 2) {
    initial << Scalar(1.5), Scalar(1.5);
  } else {
    initial << Scalar(1.5), Scalar(1.5), Scalar(1.5);
  }
  return initial;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> PriorTarget(Eigen::Index dimensions) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> target(dimensions);
  for (Eigen::Index index = 0; index < dimensions; ++index)
    target[index] = Scalar(0.25) * Scalar(index + 1);
  return target;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> PriorInitial(Eigen::Index dimensions) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> initial(dimensions);
  for (Eigen::Index index = 0; index < dimensions; ++index)
    initial[index] = Scalar(2) + Scalar(0.1) * Scalar(index);
  return initial;
}

template <typename Derived>
auto PriorResiduals(const Eigen::MatrixBase<Derived>& x) {
  using Scalar = typename Derived::Scalar;
  return x - PriorTarget<Scalar>(x.size());
}

}  // namespace tinyopt::benchmark
