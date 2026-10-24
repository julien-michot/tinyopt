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
    return 6 * problem.CameraCount() + 3 * problem.PointCount();
  }

  static std::pair<Index, Index> reduced(
      const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    return {0, 6 * problem.CameraCount()};
  }

  static std::vector<Index> locked(const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    const Index camera_offset = 6 * problem.CameraCount();
    return {0, 1, 2, 3, 4, 5, camera_offset, camera_offset + 1, camera_offset + 2};
  }

  static void PlusEq(tinyopt::benchmark::bundle_adjustment::Problem& problem, const auto& delta) {
    const int camera_offset = 6 * problem.CameraCount();
    for (int camera = 0; camera < problem.CameraCount(); ++camera)
      problem.initial_cameras[camera] += delta.template segment<6>(camera * 6);
    for (int point = 0; point < problem.PointCount(); ++point)
      problem.initial_points[point] += delta.template segment<3>(camera_offset + point * 3);
  }

  static Scalar norm2(const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    Scalar sum = 0.0;
    for (const auto& camera : problem.initial_cameras) sum += camera.squaredNorm();
    for (const auto& point : problem.initial_points) sum += point.squaredNorm();
    return sum;
  }
};

}  // namespace tinyopt::traits

namespace tinyopt::benchmark::bundle_adjustment {

struct BlockSparseBundleAdjustmentLoss {
  using Hessian = BlockSparseMatrix<double, 6, 3>;

  Cost operator()(const Problem& problem, auto& gradient, Hessian& hessian) const {
    const int camera_offset = 6 * problem.CameraCount();
    double squared_error = 0.0;

    Eigen::Matrix<double, 2, 6> camera_jacobian;
    Eigen::Matrix<double, 2, 3> point_jacobian;
    ImagePoint projected;

    for (const auto& observation : problem.observations) {
      const Camera& camera = problem.initial_cameras[observation.camera];
      const Point& point = problem.initial_points[observation.point];

      if constexpr (traits::is_nullptr_v<decltype(gradient)>) {
        const ImagePoint residual = Project(camera, point) - observation.measurement;
        squared_error += residual.squaredNorm();
      } else {
        ProjectionJacobians(camera, point, projected, camera_jacobian, point_jacobian);
        const ImagePoint residual = projected - observation.measurement;
        squared_error += residual.squaredNorm();

        const Index camera_block = observation.camera;
        const Index point_block = observation.point;
        const Index camera_start = camera_block * 6;
        const Index point_start = camera_offset + point_block * 3;

        gradient.template segment<6>(camera_start).noalias() +=
            camera_jacobian.transpose() * residual;
        gradient.template segment<3>(point_start).noalias() +=
            point_jacobian.transpose() * residual;

        hessian.template block<6, 6>(camera_block, camera_block).noalias() +=
            camera_jacobian.transpose() * camera_jacobian;
        hessian.template block<3, 3>(point_block, point_block).noalias() +=
            point_jacobian.transpose() * point_jacobian;
        hessian.template block<6, 3>(camera_block, point_block).noalias() +=
            camera_jacobian.transpose() * point_jacobian;
      }
    }
    return Cost(0.5 * squared_error, static_cast<int>(2 * problem.observations.size()));
  }
};

using BlockSparseLoss = BlockSparseBundleAdjustmentLoss;
using TinyoptLoss = BlockSparseBundleAdjustmentLoss;

struct SparseBundleAdjustmentLoss {
  using Hessian = SparseMat;

  Cost operator()(const Problem& problem, auto& gradient, Hessian& hessian) const {
    const int camera_offset = 6 * problem.CameraCount();
    double squared_error = 0.0;

    Eigen::Matrix<double, 2, 6> camera_jacobian;
    Eigen::Matrix<double, 2, 3> point_jacobian;
    ImagePoint projected;

    std::vector<Eigen::Triplet<double>> triplets;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      triplets.reserve(problem.observations.size() * (36 + 9 + 36));
    }

    for (const auto& observation : problem.observations) {
      const Camera& camera = problem.initial_cameras[observation.camera];
      const Point& point = problem.initial_points[observation.point];

      if constexpr (traits::is_nullptr_v<decltype(gradient)>) {
        const ImagePoint residual = Project(camera, point) - observation.measurement;
        squared_error += residual.squaredNorm();
      } else {
        ProjectionJacobians(camera, point, projected, camera_jacobian, point_jacobian);
        const ImagePoint residual = projected - observation.measurement;
        squared_error += residual.squaredNorm();

        const Index camera_block = observation.camera;
        const Index point_block = observation.point;
        const Index camera_start = camera_block * 6;
        const Index point_start = camera_offset + point_block * 3;

        gradient.template segment<6>(camera_start).noalias() +=
            camera_jacobian.transpose() * residual;
        gradient.template segment<3>(point_start).noalias() +=
            point_jacobian.transpose() * residual;

        const Eigen::Matrix<double, 6, 6> JtJ_cam = camera_jacobian.transpose() * camera_jacobian;
        for (int r = 0; r < 6; ++r) {
          for (int c = 0; c < 6; ++c) {
            triplets.emplace_back(camera_start + r, camera_start + c, JtJ_cam(r, c));
          }
        }

        const Eigen::Matrix<double, 3, 3> JtJ_pt = point_jacobian.transpose() * point_jacobian;
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            triplets.emplace_back(point_start + r, point_start + c, JtJ_pt(r, c));
          }
        }

        const Eigen::Matrix<double, 6, 3> JtJ_cp = camera_jacobian.transpose() * point_jacobian;
        for (int r = 0; r < 6; ++r) {
          for (int c = 0; c < 3; ++c) {
            const double val = JtJ_cp(r, c);
            triplets.emplace_back(camera_start + r, point_start + c, val);
            triplets.emplace_back(point_start + c, camera_start + r, val);
          }
        }
      }
    }

    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      hessian.setFromTriplets(triplets.begin(), triplets.end());
    }

    return Cost(0.5 * squared_error, static_cast<int>(2 * problem.observations.size()));
  }
};

using SparseLoss = SparseBundleAdjustmentLoss;

}  // namespace tinyopt::benchmark::bundle_adjustment