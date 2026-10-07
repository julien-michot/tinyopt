// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/tinyopt.h>

namespace tinyopt::benchmark {

struct DenseMathLoss {
  template <typename Parameters, typename Gradient, typename Hessian>
  Cost operator()(const Parameters& x, Gradient& gradient, Hessian& hessian) const {
    using Scalar = typename Parameters::Scalar;
    const Index dimensions = x.size();
    Eigen::Matrix<Scalar, 3, 1> residuals = Eigen::Matrix<Scalar, 3, 1>::Zero();
    Eigen::Matrix<Scalar, 3, 3> jacobian = Eigen::Matrix<Scalar, 3, 3>::Zero();

    if (dimensions == 1) {
      residuals[0] = x[0] * x[0] - Scalar(2);
      jacobian(0, 0) = Scalar(2) * x[0];
    } else if (dimensions == 2) {
      residuals[0] = x[0] * x[0] + x[1] - Scalar(2);
      residuals[1] = x[0] + x[1] * x[1] - Scalar(2);
      jacobian(0, 0) = Scalar(2) * x[0];
      jacobian(0, 1) = Scalar(1);
      jacobian(1, 0) = Scalar(1);
      jacobian(1, 1) = Scalar(2) * x[1];
    } else {
      residuals[0] = x[0] * x[0] + x[1] + x[2] - Scalar(3);
      residuals[1] = x[0] + x[1] * x[1] + x[2] - Scalar(3);
      residuals[2] = x[0] + x[1] + x[2] * x[2] - Scalar(3);
      jacobian.setOnes();
      jacobian.diagonal() = Scalar(2) * x.template head<3>();
    }

    if constexpr (!traits::is_nullptr_v<Gradient>) {
      gradient.setZero();
      for (Index column = 0; column < dimensions; ++column)
        for (Index row = 0; row < dimensions; ++row)
          gradient[column] += jacobian(row, column) * residuals[row];
    }
    if constexpr (!traits::is_nullptr_v<Hessian>) {
      hessian.setZero();
      for (Index row = 0; row < dimensions; ++row) {
        for (Index column = 0; column < dimensions; ++column) {
          for (Index residual = 0; residual < dimensions; ++residual)
            hessian(row, column) +=
                jacobian(residual, row) * jacobian(residual, column);
        }
      }
    }
    return Cost(Scalar(0.5) * residuals.head(dimensions).squaredNorm(),
                static_cast<int>(dimensions));
  }
};

}  // namespace tinyopt::benchmark
