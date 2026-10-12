// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/tinyopt.h>

#include "bundle_adjustment.h"

namespace tinyopt::traits {

template <>
struct params_trait<tinyopt::benchmark::bundle_adjustment::Problem> {
  using Scalar = double;
  static constexpr Index Dims = Dynamic;

  static Index dims(const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    return 6 * (problem.CameraCount() - 1) + 3 * (problem.PointCount() - 1);
  }

  static std::pair<Index, Index> reduced(
      const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    return {0, 6 * (problem.CameraCount() - 1)};
  }

  static void PlusEq(tinyopt::benchmark::bundle_adjustment::Problem& problem, const auto& delta) {
    const int camera_offset = 6 * (problem.CameraCount() - 1);
    for (int camera = 1; camera < problem.CameraCount(); ++camera)
      problem.initial_cameras[camera] += delta.template segment<6>((camera - 1) * 6);
    for (int point = 1; point < problem.PointCount(); ++point)
      problem.initial_points[point] += delta.template segment<3>(camera_offset + (point - 1) * 3);
  }
};

}  // namespace tinyopt::traits

namespace tinyopt::benchmark::bundle_adjustment {

using LocalParameters = Eigen::Matrix<double, 9, 1>;
using LocalJacobian = Eigen::Matrix<double, 2, 9>;
using LocalGradient = Eigen::Matrix<double, 9, 1>;
using LocalHessian = Eigen::Matrix<double, 9, 9>;
using LocalJet = diff::Jet<double, 9>;

inline LocalJacobian ProjectionJacobian(const LocalParameters& parameters,
                                        const ImagePoint& measurement) {
  Eigen::Matrix<LocalJet, 9, 1> differentiated;
  for (int index = 0; index < 9; ++index) {
    differentiated[index] = LocalJet(parameters[index]);
    differentiated[index].v.setZero();
    differentiated[index].v[index] = 1.0;
  }

  const Eigen::Matrix<LocalJet, 6, 1> camera = differentiated.template head<6>();
  const Eigen::Matrix<LocalJet, 3, 1> point = differentiated.template tail<3>();
  const Eigen::Matrix<LocalJet, 2, 1> residual =
      Project(camera, point) - measurement.cast<LocalJet>();
  LocalJacobian jacobian;
  for (int row = 0; row < 2; ++row) jacobian.row(row) = residual[row].v.transpose();
  return jacobian;
}

struct TinyoptLoss {
  using Hessian = BlockSparseMatrix<double, 6, 3>;

  Cost operator()(const Problem& problem, auto& gradient, Hessian& hessian) const {
    const int camera_offset = 6 * (problem.CameraCount() - 1);
    double squared_error = 0.0;

    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.setZero();
    }

    for (const auto& observation : problem.observations) {
      const Camera& camera = problem.initial_cameras[observation.camera];
      const Point& point = problem.initial_points[observation.point];
      const ImagePoint residual = Project(camera, point) - observation.measurement;
      squared_error += residual.squaredNorm();

      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        LocalParameters local;
        local << camera, point;
        const LocalJacobian jacobian = ProjectionJacobian(local, observation.measurement);
        const LocalGradient local_gradient = jacobian.transpose() * residual;
        const LocalHessian local_hessian = jacobian.transpose() * jacobian;
        const Index camera_block = observation.camera - 1;
        const Index point_block = observation.point - 1;
        if (camera_block >= 0) {
          const Index camera_start = camera_block * 6;
          gradient.template segment<6>(camera_start) += local_gradient.head<6>();
          hessian.template block<6, 6>(camera_block, camera_block) +=
              local_hessian.template topLeftCorner<6, 6>();
        }
        if (point_block >= 0) {
          const Index point_start = camera_offset + point_block * 3;
          gradient.template segment<3>(point_start) += local_gradient.tail<3>();
          hessian.template block<3, 3>(point_block, point_block) +=
              local_hessian.template bottomRightCorner<3, 3>();
        }
        if (camera_block >= 0 && point_block >= 0) {
          hessian.template block<6, 3>(camera_block, point_block) +=
              local_hessian.template topRightCorner<6, 3>();
        }
      }
    }
    return Cost(0.5 * squared_error, static_cast<int>(2 * problem.observations.size()));
  }
};

}  // namespace tinyopt::benchmark::bundle_adjustment