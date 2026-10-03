// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Core>

namespace tinyopt::benchmark::bundle_adjustment {

inline constexpr int CameraCount = 5;
inline constexpr int PointCount = 50;
inline constexpr int ObservationCount = CameraCount * PointCount;
inline constexpr double FocalX = 520.0;
inline constexpr double FocalY = 520.0;
inline constexpr double PrincipalX = 320.0;
inline constexpr double PrincipalY = 240.0;

using Camera = Eigen::Matrix<double, 6, 1>;
using Point = Eigen::Vector3d;
using ImagePoint = Eigen::Vector2d;

struct Observation {
  int camera;
  int point;
  ImagePoint measurement;
};

struct Problem {
  std::array<Camera, CameraCount> truth_cameras;
  std::array<Camera, CameraCount> initial_cameras;
  std::array<Point, PointCount> truth_points;
  std::array<Point, PointCount> initial_points;
  std::vector<Observation> observations;
};

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> RotateEuler(const Eigen::Matrix<Scalar, 3, 1>& angles,
                                        const Eigen::Matrix<Scalar, 3, 1>& point) {
  using std::cos;
  using std::sin;

  const Scalar cx = cos(angles[0]);
  const Scalar sx = sin(angles[0]);
  const Scalar cy = cos(angles[1]);
  const Scalar sy = sin(angles[1]);
  const Scalar cz = cos(angles[2]);
  const Scalar sz = sin(angles[2]);

  Eigen::Matrix<Scalar, 3, 1> rotated;
  const Scalar y1 = cx * point[1] - sx * point[2];
  const Scalar z1 = sx * point[1] + cx * point[2];
  const Scalar x2 = cy * point[0] + sy * z1;
  const Scalar z2 = -sy * point[0] + cy * z1;
  rotated[0] = cz * x2 - sz * y1;
  rotated[1] = sz * x2 + cz * y1;
  rotated[2] = z2;
  return rotated;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> Project(const Eigen::Matrix<Scalar, 6, 1>& camera,
                                    const Eigen::Matrix<Scalar, 3, 1>& point) {
  const Eigen::Matrix<Scalar, 3, 1> angles = camera.template head<3>();
  const Eigen::Matrix<Scalar, 3, 1> world_point = point;
  const Eigen::Matrix<Scalar, 3, 1> camera_point =
      RotateEuler(angles, world_point) + camera.template tail<3>();
  Eigen::Matrix<Scalar, 2, 1> projected;
  projected[0] = Scalar(FocalX) * camera_point[0] / camera_point[2] + Scalar(PrincipalX);
  projected[1] = Scalar(FocalY) * camera_point[1] / camera_point[2] + Scalar(PrincipalY);
  return projected;
}

inline Problem MakeProblem() {
  Problem problem;
  problem.observations.reserve(ObservationCount);

  for (int camera_index = 0; camera_index < CameraCount; ++camera_index) {
    Camera camera = Camera::Zero();
    camera[0] = 0.002 * camera_index;
    camera[1] = -0.018 * (camera_index - 2);
    camera[2] = 0.003 * camera_index;
    camera[3] = -0.18 * (camera_index - 2);
    camera[4] = 0.015 * (camera_index - 2);
    camera[5] = 0.02 * (camera_index - 2);
    problem.truth_cameras[camera_index] = camera;
    problem.initial_cameras[camera_index] = camera;
    if (camera_index != 0) {
      problem.initial_cameras[camera_index][0] += 0.006 * camera_index;
      problem.initial_cameras[camera_index][1] -= 0.004;
      problem.initial_cameras[camera_index][3] += 0.035;
      problem.initial_cameras[camera_index][4] -= 0.02 * camera_index;
      problem.initial_cameras[camera_index][5] += 0.025;
    }
  }

  for (int point_index = 0; point_index < PointCount; ++point_index) {
    const int column = point_index % 10;
    const int row = point_index / 10;
    Point point((column - 4.5) * 0.24, (row - 2.0) * 0.27, 4.5 + 0.11 * (point_index % 7));
    problem.truth_points[point_index] = point;
    problem.initial_points[point_index] = point;
    if (point_index != 0) {
      problem.initial_points[point_index] += Point(0.02, -0.015, 0.025);
    }
  }

  for (int camera_index = 0; camera_index < CameraCount; ++camera_index) {
    for (int point_index = 0; point_index < PointCount; ++point_index) {
      problem.observations.push_back(
          {camera_index, point_index,
           Project(problem.truth_cameras[camera_index], problem.truth_points[point_index])});
    }
  }
  return problem;
}

inline double ReferenceReprojectionCost(const Problem& problem) {
  double squared_error = 0.0;
  for (const auto& observation : problem.observations) {
    const ImagePoint residual = Project(problem.initial_cameras[observation.camera],
                                        problem.initial_points[observation.point]) -
                                observation.measurement;
    squared_error += residual.squaredNorm();
  }
  return 0.5 * squared_error;
}

inline Eigen::VectorXd PackInitial(const Problem& problem) {
  constexpr int CameraOffset = 6 * (CameraCount - 1);
  Eigen::VectorXd parameters(CameraOffset + 3 * (PointCount - 1));
  for (int camera_index = 1; camera_index < CameraCount; ++camera_index) {
    parameters.segment<6>((camera_index - 1) * 6) = problem.initial_cameras[camera_index];
  }
  for (int point_index = 1; point_index < PointCount; ++point_index) {
    parameters.segment<3>(CameraOffset + (point_index - 1) * 3) =
        problem.initial_points[point_index];
  }
  return parameters;
}

}  // namespace tinyopt::benchmark::bundle_adjustment