// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

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
  Cost operator()(const Problem& problem, auto& gradient, SparseMat& hessian) const {
    const int camera_offset = 6 * (problem.CameraCount() - 1);
    double squared_error = 0.0;

    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.setZero();
      hessian_entries_.clear();
      hessian_entries_.reserve(81 * problem.observations.size());
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
        std::array<Index, 9> global_indices;
        global_indices.fill(-1);
        if (observation.camera != 0) {
          for (int index = 0; index < 6; ++index)
            global_indices[index] = (observation.camera - 1) * 6 + index;
        }
        if (observation.point != 0) {
          for (int index = 0; index < 3; ++index)
            global_indices[6 + index] = camera_offset + (observation.point - 1) * 3 + index;
        }

        const Index camera_start = global_indices[0];
        const Index point_start = global_indices[6];
        if (camera_start >= 0)
          gradient.template segment<6>(camera_start) += local_gradient.head<6>();
        if (point_start >= 0) gradient.template segment<3>(point_start) += local_gradient.tail<3>();
        for (int column = 0; column < 9; ++column) {
          const Index global_column = global_indices[column];
          if (global_column < 0) continue;
          for (int row = 0; row < 9; ++row) {
            const Index global_row = global_indices[row];
            if (global_row >= 0)
              hessian_entries_.emplace_back(static_cast<SparseMat::StorageIndex>(global_row),
                                            static_cast<SparseMat::StorageIndex>(global_column),
                                            local_hessian(row, column));
          }
        }
      }
    }
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>)
      hessian.setFromTriplets(hessian_entries_.begin(), hessian_entries_.end());
    return Cost(0.5 * squared_error, static_cast<int>(2 * problem.observations.size()));
  }

 private:
  mutable std::vector<Eigen::Triplet<double>> hessian_entries_;
};

}  // namespace tinyopt::benchmark::bundle_adjustment