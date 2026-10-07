// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <random>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace tinyopt::benchmark::robust_pose {

using Pose = Sophus::SE3d;
using Point = Eigen::Vector3d;

struct Observation {
  Point world_point;
  Point measurement;
};

inline Pose GroundTruth() {
  Eigen::Matrix<double, 6, 1> tangent;
  tangent << 0.15, -0.1, 0.2, 0.3, -0.2, 0.1;
  return Pose::exp(tangent);
}

inline Pose InitialPose(const Pose& ground_truth) {
  Eigen::Matrix<double, 6, 1> perturbation;
  perturbation << 0.05, -0.04, 0.06, 0.07, -0.06, 0.04;
  return ground_truth * Pose::exp(perturbation);
}

inline std::vector<Observation> MakeObservations(const Pose& ground_truth, int inliers,
                                                 int outliers, double inlier_sigma,
                                                 double outlier_sigma, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> inlier_noise(0.0, inlier_sigma);
  std::normal_distribution<double> outlier_noise(0.0, outlier_sigma);
  std::uniform_real_distribution<double> coordinate(-5.0, 5.0);

  std::vector<Observation> observations;
  observations.reserve(inliers + outliers);
  for (int index = 0; index < inliers; ++index) {
    const Point point(coordinate(rng), coordinate(rng), coordinate(rng));
    const Point noise(inlier_noise(rng), inlier_noise(rng), inlier_noise(rng));
    observations.push_back({point, ground_truth * point + noise});
  }
  for (int index = 0; index < outliers; ++index) {
    const Point point(coordinate(rng), coordinate(rng), coordinate(rng));
    const Point noise(outlier_noise(rng), outlier_noise(rng), outlier_noise(rng));
    observations.push_back({point, ground_truth * point + noise});
  }
  return observations;
}

}  // namespace tinyopt::benchmark::robust_pose
