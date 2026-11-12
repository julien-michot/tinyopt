// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ceres/ceres.h>

#include "bundle_adjustment.h"

namespace tinyopt::benchmark::bundle_adjustment {

/// SE(3) manifold with the same right update as Tinyopt: T <- T * Exp(delta).
class SE3Manifold final : public ceres::Manifold {
 public:
  int AmbientSize() const override { return Camera::Ambient; }
  int TangentSize() const override { return Camera::Tangent; }

  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
    Camera camera;
    std::copy(x, x + Camera::Ambient, camera.data());
    camera.plus_eq(Eigen::Map<const Vec6>(delta));
    std::copy(camera.data(), camera.data() + Camera::Ambient, x_plus_delta);
    return true;
  }

  bool PlusJacobian(const double* x, double* jacobian) const override {
    Camera camera;
    std::copy(x, x + Camera::Ambient, camera.data());
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
    J = camera.PlusJacobian();
    return true;
  }

  bool Minus(const double* y, const double* x, double* y_minus_x) const override {
    Camera from, to;
    std::copy(x, x + Camera::Ambient, from.data());
    std::copy(y, y + Camera::Ambient, to.data());
    Eigen::Map<Vec6> delta(y_minus_x);
    delta = from.Minus(to);
    return true;
  }

  bool MinusJacobian(const double* x, double* jacobian) const override {
    Camera camera;
    std::copy(x, x + Camera::Ambient, camera.data());
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
    J = camera.MinusJacobian();
    return true;
  }
};

/// Reprojection residual; its camera Jacobian is expressed in the ambient space so that Ceres'
/// chain rule through PlusJacobian recovers the exact tangent-space Jacobian.
struct ReprojectionCost final : ceres::SizedCostFunction<2, 7, 3> {
  explicit ReprojectionCost(const ImagePoint& measurement) : measurement_(measurement) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    Camera camera;
    std::copy(parameters[0], parameters[0] + Camera::Ambient, camera.data());
    const Eigen::Map<const Point> point(parameters[1]);
    Eigen::Matrix<double, 2, 6> camera_jacobian;
    Eigen::Matrix<double, 2, 3> point_jacobian;
    ImagePoint projected;
    ProjectionJacobians(camera, point, projected, camera_jacobian, point_jacobian);
    Eigen::Map<ImagePoint> output(residuals);
    output = projected - measurement_;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[0]);
      J.noalias() = camera_jacobian * camera.MinusJacobian();
    }
    if (jacobians != nullptr && jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
      J = point_jacobian;
    }
    return true;
  }

  ImagePoint measurement_;
};

}  // namespace tinyopt::benchmark::bundle_adjustment
