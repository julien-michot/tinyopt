// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "iterations.h"

namespace tinyopt::benchmark::bundle_adjustment {

inline constexpr double FocalX = 520.0;
inline constexpr double FocalY = 520.0;
inline constexpr double PrincipalX = 320.0;
inline constexpr double PrincipalY = 240.0;
inline constexpr int PointCameraWindow = 5;

using Camera = Eigen::Matrix<double, 6, 1>;
using Point = Eigen::Vector3d;
using ImagePoint = Eigen::Vector2d;

struct Observation {
  int camera;
  int point;
  ImagePoint measurement;
};

struct Problem {
  std::vector<Camera> truth_cameras;
  std::vector<Camera> initial_cameras;
  std::vector<Point> truth_points;
  std::vector<Point> initial_points;
  std::vector<Observation> observations;

  int CameraCount() const { return static_cast<int>(truth_cameras.size()); }
  int PointCount() const { return static_cast<int>(truth_points.size()); }
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
  const Eigen::Matrix<Scalar, 3, 1> camera_point =
      RotateEuler(angles, point) + camera.template tail<3>();
  Eigen::Matrix<Scalar, 2, 1> projected;
  projected[0] = Scalar(FocalX) * camera_point[0] / camera_point[2] + Scalar(PrincipalX);
  projected[1] = Scalar(FocalY) * camera_point[1] / camera_point[2] + Scalar(PrincipalY);
  return projected;
}

inline void ProjectionJacobians(const Camera& camera, const Point& point, ImagePoint& projected,
                               Eigen::Matrix<double, 2, 6>& camera_jacobian,
                               Eigen::Matrix<double, 2, 3>& point_jacobian) {
  const double cx = std::cos(camera[0]);
  const double sx = std::sin(camera[0]);
  const double cy = std::cos(camera[1]);
  const double sy = std::sin(camera[1]);
  const double cz = std::cos(camera[2]);
  const double sz = std::sin(camera[2]);

  Eigen::Matrix3d rx, ry, rz, drx, dry, drz;
  rx << 1, 0, 0, 0, cx, -sx, 0, sx, cx;
  ry << cy, 0, sy, 0, 1, 0, -sy, 0, cy;
  rz << cz, -sz, 0, sz, cz, 0, 0, 0, 1;
  drx << 0, 0, 0, 0, -sx, -cx, 0, cx, -sx;
  dry << -sy, 0, cy, 0, 0, 0, -cy, 0, -sy;
  drz << -sz, -cz, 0, cz, -sz, 0, 0, 0, 0;

  const Eigen::Matrix3d rotation = rz * ry * rx;
  const Eigen::Vector3d camera_point = rotation * point + camera.tail<3>();
  const double x = camera_point[0];
  const double y = camera_point[1];
  const double z = camera_point[2];
  projected << FocalX * x / z + PrincipalX, FocalY * y / z + PrincipalY;

  Eigen::Matrix<double, 2, 3> projection_jacobian;
  projection_jacobian << FocalX / z, 0, -FocalX * x / (z * z), 0, FocalY / z,
      -FocalY * y / (z * z);
  Eigen::Matrix<double, 3, 3> rotation_jacobian;
  rotation_jacobian.col(0) = rz * ry * drx * point;
  rotation_jacobian.col(1) = rz * dry * rx * point;
  rotation_jacobian.col(2) = drz * ry * rx * point;
  camera_jacobian.leftCols<3>().noalias() = projection_jacobian * rotation_jacobian;
  camera_jacobian.rightCols<3>() = projection_jacobian;
  point_jacobian.noalias() = projection_jacobian * rotation;
}

inline Problem MakeProblem(int camera_count, int point_count) {
  if (camera_count < 2 || point_count < 2)
    throw std::invalid_argument("Bundle adjustment requires at least two cameras and points");

  Problem problem;
  problem.truth_cameras.resize(camera_count);
  problem.initial_cameras.resize(camera_count);
  problem.truth_points.resize(point_count);
  problem.initial_points.resize(point_count);
  problem.observations.reserve(
      static_cast<std::size_t>(point_count) *
      static_cast<std::size_t>(std::min(camera_count - 1, PointCameraWindow)));

  for (int camera_index = 0; camera_index < camera_count; ++camera_index) {
    Camera camera = Camera::Zero();
    camera[0] = 0.002 * camera_index;
    camera[1] = -0.018 * (camera_index - (camera_count - 1) * 0.5);
    camera[2] = 0.003 * camera_index;
    camera[3] = -0.18 * (camera_index - (camera_count - 1) * 0.5);
    camera[4] = 0.015 * (camera_index - (camera_count - 1) * 0.5);
    camera[5] = 0.02 * (camera_index - (camera_count - 1) * 0.5);
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

  for (int point_index = 0; point_index < point_count; ++point_index) {
    const int column = point_index % 10;
    const int row = point_index / 10;
    const Point point((column - 4.5) * 0.24, (row % 20 - 9.5) * 0.12,
                      4.5 + 0.11 * (point_index % 7));
    problem.truth_points[point_index] = point;
    problem.initial_points[point_index] = point;
    if (point_index != 0)
      problem.initial_points[point_index] += Point(0.02, -0.015, 0.025);
  }

  const int window = std::min(camera_count - 1, PointCameraWindow);
  const int start_count = camera_count - window + 1;
  for (int point_index = 0; point_index < point_count; ++point_index) {
    const int first_camera = point_index % start_count;
    for (int offset = 0; offset < window; ++offset) {
      const int camera_index = first_camera + offset;
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

inline std::string ProblemLabel(const Problem& problem) {
  return std::to_string(problem.CameraCount()) + "c " + std::to_string(problem.PointCount()) +
         "p";
}

}  // namespace tinyopt::benchmark::bundle_adjustment
