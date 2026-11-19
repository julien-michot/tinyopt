// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>

#include <tinyopt/optimize_factors_sycl.h>

namespace tinyopt::benchmark::bundle_adjustment_sycl {

inline constexpr int Dims = 9;
inline constexpr int PointCount = 3;
inline constexpr std::size_t ObservationCount = 2 * PointCount;
inline constexpr Index FactorCount = 16384;
inline constexpr double FocalLength = 520.0;

using Parameters = Vector<double, Dims>;
using Contribution = SyclFactorContribution<double, Dims>;

struct Observation {
  int camera;
  int point;
  double measurement_x;
  double measurement_y;
};

struct Factor {
  std::array<Observation, ObservationCount> observations{};
  std::array<double, 3> anchor_point{};

  Contribution operator()(Index index, const std::array<double, Dims>& value) const {
    const Observation& observation =
        observations[static_cast<std::size_t>(index) % ObservationCount];
    std::array<double, 3> point{};
    if (observation.point == 0) {
      point = anchor_point;
    } else {
      const std::size_t point_offset =
          static_cast<std::size_t>(3 + (observation.point - 1) * 3);
      for (std::size_t axis = 0; axis < 3; ++axis) point[axis] = value[point_offset + axis];
    }

    if (observation.camera == 1) {
      for (std::size_t axis = 0; axis < 3; ++axis) point[axis] += value[axis];
    }

    const double inverse_z = 1.0 / point[2];
    const double inverse_z_squared = inverse_z * inverse_z;
    const double projected_x = FocalLength * point[0] * inverse_z;
    const double projected_y = FocalLength * point[1] * inverse_z;
    const double residual_x = projected_x - observation.measurement_x;
    const double residual_y = projected_y - observation.measurement_y;
    const std::array<double, 3> jacobian_x{
        FocalLength * inverse_z, 0.0, -FocalLength * point[0] * inverse_z_squared};
    const std::array<double, 3> jacobian_y{
        0.0, FocalLength * inverse_z, -FocalLength * point[1] * inverse_z_squared};

    std::array<double, Dims> full_jacobian_x{};
    std::array<double, Dims> full_jacobian_y{};
    if (observation.camera == 1) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        full_jacobian_x[axis] = jacobian_x[axis];
        full_jacobian_y[axis] = jacobian_y[axis];
      }
    }
    if (observation.point != 0) {
      const std::size_t point_offset =
          static_cast<std::size_t>(3 + (observation.point - 1) * 3);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        full_jacobian_x[point_offset + axis] = jacobian_x[axis];
        full_jacobian_y[point_offset + axis] = jacobian_y[axis];
      }
    }

    Contribution contribution;
    contribution.cost = 0.5 * (residual_x * residual_x + residual_y * residual_y);
    contribution.num_residuals = 2;
    for (int row = 0; row < Dims; ++row) {
      const std::size_t row_index = static_cast<std::size_t>(row);
      contribution.gradient[row_index] =
          full_jacobian_x[row_index] * residual_x + full_jacobian_y[row_index] * residual_y;
      for (int col = 0; col < Dims; ++col) {
        const std::size_t col_index = static_cast<std::size_t>(col);
        contribution.hessian[row_index * Dims + col_index] =
            full_jacobian_x[row_index] * full_jacobian_x[col_index] +
            full_jacobian_y[row_index] * full_jacobian_y[col_index];
      }
    }
    return contribution;
  }
};

inline Parameters GroundTruth() {
  Parameters value;
  value << 0.75, -0.02, 0.1, 0.4, 0.3, 4.7, -0.4, 0.2, 5.2;
  return value;
}

inline Parameters InitialValue() {
  Parameters value;
  value << 0.5, -0.1, 0.2, 0.43, 0.27, 4.75, -0.36, 0.17, 5.15;
  return value;
}

inline std::array<double, 3> AnchorPoint() { return {-0.3, -0.2, 4.2}; }

inline Factor MakeFactor() {
  const Parameters truth = GroundTruth();
  const auto anchor = AnchorPoint();
  Factor factor;
  factor.anchor_point = anchor;
  for (int camera = 0; camera < 2; ++camera) {
    for (int point = 0; point < PointCount; ++point) {
      std::array<double, 3> position{};
      if (point == 0) {
        position = anchor;
      } else {
        const int offset = 3 + (point - 1) * 3;
        for (int axis = 0; axis < 3; ++axis)
          position[static_cast<std::size_t>(axis)] = truth[offset + axis];
      }
      if (camera == 1) {
        for (int axis = 0; axis < 3; ++axis)
          position[static_cast<std::size_t>(axis)] += truth[axis];
      }
      const double measurement_x = FocalLength * position[0] / position[2];
      const double measurement_y = FocalLength * position[1] / position[2];
      factor.observations[static_cast<std::size_t>(camera * PointCount + point)] =
          {camera, point, measurement_x, measurement_y};
    }
  }
  return factor;
}

inline Cost AccumulateCpuFactor(const Factor& factor, Index index, const Parameters& value,
                                Parameters& gradient, Matrix<double, Dims, Dims>& hessian) {
  std::array<double, Dims> parameters{};
  for (int i = 0; i < Dims; ++i) parameters[static_cast<std::size_t>(i)] = value[i];
  const Contribution contribution = factor(index, parameters);
  for (int row = 0; row < Dims; ++row) {
    const std::size_t row_index = static_cast<std::size_t>(row);
    gradient[row] += contribution.gradient[row_index];
    for (int col = 0; col < Dims; ++col) {
      const std::size_t col_index = static_cast<std::size_t>(col);
      hessian(row, col) += contribution.hessian[row_index * Dims + col_index];
    }
  }
  return Cost(contribution.cost, contribution.num_residuals);
}

}  // namespace tinyopt::benchmark::bundle_adjustment_sycl
