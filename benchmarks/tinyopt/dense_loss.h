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
    Eigen::Matrix<Scalar, 3, 1> parameters = Eigen::Matrix<Scalar, 3, 1>::Zero();
    Eigen::Matrix<Scalar, 3, 1> target = Eigen::Matrix<Scalar, 3, 1>::Zero();
    Eigen::Matrix<Scalar, 3, 3> coupling = Eigen::Matrix<Scalar, 3, 3>::Zero();
    if constexpr (Parameters::RowsAtCompileTime == 1) {
      parameters[0] = x[0];
      target[0] = Scalar(2);
    } else if constexpr (Parameters::RowsAtCompileTime == 2) {
      parameters.template head<2>() = x;
      target.template head<2>().setConstant(Scalar(2));
      coupling.template topLeftCorner<2, 2>().setOnes();
      coupling.diagonal().template head<2>().setZero();
    } else if constexpr (Parameters::RowsAtCompileTime == 3) {
      parameters = x;
      target.template head<3>().setConstant(Scalar(3));
      coupling.setOnes();
      coupling.diagonal().setZero();
    } else {
      if (dimensions == 1) {
        parameters[0] = x[0];
        target[0] = Scalar(2);
      } else if (dimensions == 2) {
        parameters.template head<2>() = x.template head<2>();
        target.template head<2>().setConstant(Scalar(2));
        coupling.template topLeftCorner<2, 2>().setOnes();
        coupling.diagonal().template head<2>().setZero();
      } else {
        parameters = x;
        target.template head<3>().setConstant(Scalar(3));
        coupling.setOnes();
        coupling.diagonal().setZero();
      }
    }

    const Eigen::Matrix<Scalar, 3, 1> residuals =
        parameters.array().square().matrix() + coupling * parameters - target;
    Eigen::Matrix<Scalar, 3, 3> jacobian = coupling;
    jacobian.diagonal() += Scalar(2) * parameters;

    if constexpr (!traits::is_nullptr_v<Gradient>) {
      const Eigen::Matrix<Scalar, 3, 1> dense_gradient = jacobian.transpose() * residuals;
      if constexpr (Parameters::RowsAtCompileTime == 1) {
        gradient[0] = dense_gradient[0];
      } else if constexpr (Parameters::RowsAtCompileTime == 2) {
        gradient.template head<2>() = dense_gradient.template head<2>();
      } else if constexpr (Parameters::RowsAtCompileTime == 3) {
        gradient = dense_gradient;
      } else if (dimensions == 1) {
        gradient[0] = dense_gradient[0];
      } else if (dimensions == 2) {
        gradient.template head<2>() = dense_gradient.template head<2>();
      } else {
        gradient.template head<3>() = dense_gradient;
      }
      if constexpr (!traits::is_nullptr_v<Hessian>) {
        const Eigen::Matrix<Scalar, 3, 3> dense_hessian = jacobian.transpose() * jacobian;
        if constexpr (Parameters::RowsAtCompileTime == 1) {
          hessian(0, 0) = dense_hessian(0, 0);
        } else if constexpr (Parameters::RowsAtCompileTime == 2) {
          hessian.template topLeftCorner<2, 2>() = dense_hessian.template topLeftCorner<2, 2>();
        } else if constexpr (Parameters::RowsAtCompileTime == 3) {
          hessian = dense_hessian;
        } else if (dimensions == 1) {
          hessian(0, 0) = dense_hessian(0, 0);
        } else if (dimensions == 2) {
          hessian.template topLeftCorner<2, 2>() = dense_hessian.template topLeftCorner<2, 2>();
        } else {
          hessian.template topLeftCorner<3, 3>() = dense_hessian;
        }
      }
    }
    Scalar squared_error;
    if constexpr (Parameters::RowsAtCompileTime == 1) {
      squared_error = residuals[0] * residuals[0];
    } else if constexpr (Parameters::RowsAtCompileTime == 2) {
      squared_error = residuals.template head<2>().squaredNorm();
    } else if constexpr (Parameters::RowsAtCompileTime == 3) {
      squared_error = residuals.squaredNorm();
    } else if (dimensions == 1) {
      squared_error = residuals[0] * residuals[0];
    } else if (dimensions == 2) {
      squared_error = residuals.template head<2>().squaredNorm();
    } else {
      squared_error = residuals.squaredNorm();
    }
    return Cost(Scalar(0.5) * squared_error, static_cast<int>(dimensions));
  }
};

}  // namespace tinyopt::benchmark
