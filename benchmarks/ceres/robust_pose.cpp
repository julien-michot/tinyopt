// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ceres/ceres.h>

#include "robust_pose.h"
#include "iterations.h"

namespace {

struct PointResidual final : ceres::SizedCostFunction<3, 4, 3> {
  explicit PointResidual(const tinyopt::benchmark::robust_pose::Observation& observation)
      : world_point(observation.world_point), measurement(observation.measurement) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const Eigen::Map<const Eigen::Vector4d> coefficients(parameters[0]);
    const Eigen::Map<const Eigen::Vector3d> translation(parameters[1]);
    const Eigen::Vector3d vector = coefficients.head<3>();
    const double scalar = coefficients[3];
    const Eigen::Quaterniond rotation(scalar, vector[0], vector[1], vector[2]);
    const Eigen::Vector3d rotated = rotation * world_point;
    Eigen::Map<Eigen::Vector3d> output(residuals);
    output = rotated + translation - measurement;

    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Matrix3d point_skew;
      point_skew << 0, -world_point[2], world_point[1], world_point[2], 0, -world_point[0],
          -world_point[1], world_point[0], 0;
      Eigen::Matrix3d vector_jacobian =
          -2.0 * world_point * vector.transpose() +
          2.0 * vector.dot(world_point) * Eigen::Matrix3d::Identity() +
          2.0 * vector * world_point.transpose() - 2.0 * scalar * point_skew;
      Eigen::Matrix<double, 3, 4> quaternion_jacobian;
      quaternion_jacobian.leftCols<3>() = vector_jacobian;
      quaternion_jacobian.col(3) = 2.0 * scalar * world_point + 2.0 * vector.cross(world_point);
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[0]);
      J = quaternion_jacobian;
    }
    if (jacobians != nullptr && jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[1]);
      J.setIdentity();
    }
    return true;
  }

  Eigen::Vector3d world_point;
  Eigen::Vector3d measurement;
};

struct Result {
  double initial_cost;
  double final_cost;
  bool usable;
  ceres::TerminationType termination;
  int iterations;
};

Result Solve(const std::vector<tinyopt::benchmark::robust_pose::Observation>& observations,
             const tinyopt::benchmark::robust_pose::Pose& initial_pose, bool robust) {
  Eigen::Quaterniond rotation = initial_pose.unit_quaternion();
  Eigen::Vector3d translation = initial_pose.translation();
  ceres::Problem problem;
  problem.AddParameterBlock(rotation.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
  problem.AddParameterBlock(translation.data(), 3);
  for (const auto& observation : observations) {
    auto* cost = new PointResidual(observation);
    ceres::LossFunction* loss =
        robust ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(0.3)) : nullptr;
    problem.AddResidualBlock(cost, loss, rotation.coeffs().data(), translation.data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 100;
  options.max_num_consecutive_invalid_steps = 20;
  options.num_threads = 1;
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-9;
  options.parameter_tolerance = 1e-8;
  options.min_relative_decrease = 1e-12;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return {summary.initial_cost, summary.final_cost, summary.IsSolutionUsable(),
          summary.termination_type, static_cast<int>(summary.iterations.size()) - 1};
}

void BenchmarkPose(int observation_count, bool robust) {
  using namespace tinyopt::benchmark::robust_pose;
  const Pose ground_truth = GroundTruth();
  const int outliers = observation_count / 4;
  const auto observations =
      MakeObservations(ground_truth, observation_count - outliers, outliers, 0.01, 50.0);
  const Result verification = Solve(observations, InitialPose(ground_truth), robust);
  REQUIRE(verification.usable);
  REQUIRE(verification.termination == ceres::CONVERGENCE);
  REQUIRE(verification.final_cost < verification.initial_cost);
  const std::string name = std::to_string(observation_count) + " obs " + (robust ? "Huber" : "L2");
  tinyopt::benchmark::PrintIterations(
      "Robust pose", std::to_string(observation_count) + "o " + (robust ? "Huber" : "L2"),
      "ceres", verification.iterations, verification.termination == ceres::CONVERGENCE);
  BENCHMARK(std::string(name)) {
    return Solve(observations, InitialPose(ground_truth), robust).final_cost;
  };
}

}  // namespace

TEST_CASE("PoseOptimization", "[benchmark][robust][pose][ceres]") {
  for (const int count : {20, 50, 100}) {
    BenchmarkPose(count, false);
    BenchmarkPose(count, true);
  }
}
