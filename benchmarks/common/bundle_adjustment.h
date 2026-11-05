// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "iterations.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace tinyopt::benchmark::bundle_adjustment {

/// Pixel noise standard deviation added to the observations by default.
inline constexpr double NoiseSigma = 0.5;
/// Relative cost decrease below which every BA benchmark solver stops. The 1e-6 default only
/// triggers after a long tail where the cost already changes by less than 0.03%.
inline constexpr double RelativeCostDecreaseTol = 1e-4;
inline constexpr double FocalX = 520.0;
inline constexpr double FocalY = 520.0;
inline constexpr double PrincipalX = 320.0;
inline constexpr double PrincipalY = 240.0;
inline constexpr int PointCameraWindow = 5;

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Point = Eigen::Vector3d;
using ImagePoint = Eigen::Vector2d;

/// World-to-camera SE(3) pose stored as [qx, qy, qz, qw, tx, ty, tz] (rotation as a unit
/// quaternion, no redundant rotation matrix). The 6D tangent is [w, v] (rotation, translation) with
/// the right update T <- T * Exp([w, v]), where Exp is the exact SE(3) exponential map.
struct Camera {
  Eigen::Matrix<double, 7, 1> params =
      (Eigen::Matrix<double, 7, 1>() << 0, 0, 0, 1, 0, 0, 0).finished();

  static constexpr int Ambient = 7;
  static constexpr int Tangent = 6;

  double* data() { return params.data(); }
  const double* data() const { return params.data(); }

  Eigen::Quaterniond rotation() const { return Eigen::Map<const Eigen::Quaterniond>(data()); }
  Eigen::Vector3d translation() const { return params.tail<3>(); }

  static Eigen::Matrix3d Skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d m;
    m << 0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0;
    return m;
  }

  /// Left Jacobian of SO(3): translation part of the SE(3) exponential.
  static Eigen::Matrix3d LeftJacobian(const Eigen::Vector3d& w) {
    const double theta2 = w.squaredNorm();
    const Eigen::Matrix3d W = Skew(w);
    double a, b;
    if (theta2 < 1e-12) {
      a = 0.5 - theta2 / 24.0;
      b = 1.0 / 6.0 - theta2 / 120.0;
    } else {
      const double theta = std::sqrt(theta2);
      a = (1.0 - std::cos(theta)) / theta2;
      b = (theta - std::sin(theta)) / (theta2 * theta);
    }
    return Eigen::Matrix3d::Identity() + a * W + b * W * W;
  }

  static Eigen::Matrix3d InverseLeftJacobian(const Eigen::Vector3d& w) {
    const double theta2 = w.squaredNorm();
    const Eigen::Matrix3d W = Skew(w);
    double c;
    if (theta2 < 1e-12) {
      c = 1.0 / 12.0 + theta2 / 720.0;
    } else {
      const double theta = std::sqrt(theta2);
      c = (1.0 - 0.5 * theta * std::sin(theta) / (1.0 - std::cos(theta))) / theta2;
    }
    return Eigen::Matrix3d::Identity() - 0.5 * W + c * W * W;
  }

  static Camera FromRotationVector(const Eigen::Vector3d& rvec, const Eigen::Vector3d& t) {
    Camera camera;
    const double angle = rvec.norm();
    const Eigen::Quaterniond q = angle < 1e-12
                                     ? Eigen::Quaterniond::Identity()
                                     : Eigen::Quaterniond(Eigen::AngleAxisd(angle, rvec / angle));
    camera.params << q.coeffs(), t;
    return camera;
  }

  static Camera Exp(const Vec6& delta) {
    return FromRotationVector(delta.head<3>(), LeftJacobian(delta.head<3>()) * delta.tail<3>());
  }

  /// this * Exp(delta)
  void PlusEq(const Vec6& delta) {
    const Eigen::Quaterniond q = rotation();
    const Eigen::Vector3d t = translation();
    const Camera inc = Exp(delta);
    Eigen::Quaterniond q_new = q * inc.rotation();
    q_new.normalize();
    params << q_new.coeffs(), t + q * inc.translation();
  }

  /// Log(this^-1 * other)
  Vec6 Minus(const Camera& other) const {
    const Eigen::Quaterniond dq = rotation().conjugate() * other.rotation();
    const Eigen::AngleAxisd aa(dq);
    Eigen::Vector3d w = aa.angle() * aa.axis();
    if (aa.angle() > M_PI) w = (aa.angle() - 2.0 * M_PI) * aa.axis();
    const Eigen::Vector3d dt = rotation().conjugate() * (other.translation() - translation());
    Vec6 delta;
    delta << w, InverseLeftJacobian(w) * dt;
    return delta;
  }

  /// d(ambient)/d(tangent) of T * Exp(delta) at delta = 0.
  Eigen::Matrix<double, 7, 6> PlusJacobian() const {
    const Eigen::Quaterniond q = rotation();
    Eigen::Matrix<double, 7, 6> J = Eigen::Matrix<double, 7, 6>::Zero();
    for (int axis = 0; axis < 3; ++axis) {
      Eigen::Quaterniond e(0, 0, 0, 0);
      e.coeffs()[axis] = 1.0;
      J.col(axis).head<4>() = 0.5 * (q * e).coeffs();
    }
    J.block<3, 3>(4, 3) = q.toRotationMatrix();
    return J;
  }

  /// Left inverse of PlusJacobian (the Jacobian of Minus at identity).
  Eigen::Matrix<double, 6, 7> MinusJacobian() const {
    const Eigen::Matrix<double, 7, 6> J = PlusJacobian();
    Eigen::Matrix<double, 6, 7> M = Eigen::Matrix<double, 6, 7>::Zero();
    M.block<3, 4>(0, 0) = 4.0 * J.block<4, 3>(0, 0).transpose();
    M.block<3, 3>(3, 4) = J.block<3, 3>(4, 3).transpose();
    return M;
  }

  bool operator==(const Camera& other) const { return params == other.params; }
};

static_assert(sizeof(Camera) == 7 * sizeof(double), "Camera must be 7 contiguous doubles");

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

inline ImagePoint Project(const Camera& camera, const Point& point) {
  const Eigen::Vector3d camera_point = camera.rotation() * point + camera.translation();
  return {FocalX * camera_point[0] / camera_point[2] + PrincipalX,
          FocalY * camera_point[1] / camera_point[2] + PrincipalY};
}

/// Projection and Jacobians w.r.t. the camera tangent ([w, v], right update) and the point.
inline void ProjectionJacobians(const Camera& camera, const Point& point, ImagePoint& projected,
                                Eigen::Matrix<double, 2, 6>& camera_jacobian,
                                Eigen::Matrix<double, 2, 3>& point_jacobian) {
  const Eigen::Matrix3d rotation = camera.rotation().toRotationMatrix();
  const Eigen::Vector3d camera_point = rotation * point + camera.translation();
  const double x = camera_point[0];
  const double y = camera_point[1];
  const double z = camera_point[2];
  projected << FocalX * x / z + PrincipalX, FocalY * y / z + PrincipalY;

  Eigen::Matrix<double, 2, 3> projection_jacobian;
  projection_jacobian << FocalX / z, 0, -FocalX * x / (z * z), 0, FocalY / z, -FocalY * y / (z * z);
  camera_jacobian.leftCols<3>().noalias() = projection_jacobian * (-rotation * Camera::Skew(point));
  camera_jacobian.rightCols<3>().noalias() = projection_jacobian * rotation;
  point_jacobian.noalias() = projection_jacobian * rotation;
}

inline Problem MakeProblem(int camera_count, int point_count, double noise_sigma = NoiseSigma,
                           unsigned int seed = 42) {
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
    const double centered = camera_index - (camera_count - 1) * 0.5;
    const Camera camera =
        Camera::FromRotationVector({0.002 * camera_index, -0.018 * centered, 0.003 * camera_index},
                                   {-0.18 * centered, 0.015 * centered, 0.02 * centered});
    problem.truth_cameras[camera_index] = camera;
    problem.initial_cameras[camera_index] = camera;
    if (camera_index != 0) {
      Vec6 perturbation;
      perturbation << 0.006 * camera_index, -0.004, 0.0, 0.035, -0.02 * camera_index, 0.025;
      problem.initial_cameras[camera_index].PlusEq(perturbation);
    }
  }

  for (int point_index = 0; point_index < point_count; ++point_index) {
    const int column = point_index % 10;
    const int row = point_index / 10;
    const Point point((column - 4.5) * 0.24, (row % 20 - 9.5) * 0.12,
                      4.5 + 0.11 * (point_index % 7));
    problem.truth_points[point_index] = point;
    problem.initial_points[point_index] = point;
    if (point_index != 0) problem.initial_points[point_index] += Point(0.02, -0.015, 0.025);
  }

  const int window = std::min(camera_count, PointCameraWindow);
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

  if (noise_sigma > 0.0) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, noise_sigma);
    for (auto& observation : problem.observations) {
      observation.measurement[0] += dist(gen);
      observation.measurement[1] += dist(gen);
    }
  }

  return problem;
}

/// Cost (0.5 * sum of squared residuals) expected at the optimum: sigma^2 per observation, with
/// 10% margin. The fit absorbs some noise, so the real minimum is slightly lower.
inline double MaxFinalCost(const Problem& problem) {
  return 1.1 * NoiseSigma * NoiseSigma * static_cast<double>(problem.observations.size());
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
  return std::to_string(problem.CameraCount()) + "c " + std::to_string(problem.PointCount()) + "p";
}

}  // namespace tinyopt::benchmark::bundle_adjustment
