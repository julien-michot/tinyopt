// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "robust_pose.h"
#include "iterations.h"

namespace {

gtsam::Key PoseKey() { return gtsam::Symbol('x', 0); }

gtsam::Pose3 ToPose(const tinyopt::benchmark::robust_pose::Pose& pose) {
  return {gtsam::Rot3(pose.unit_quaternion().toRotationMatrix()), pose.translation()};
}

class PointResidual final : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  PointResidual(gtsam::Key key, const tinyopt::benchmark::robust_pose::Observation& observation,
                const gtsam::SharedNoiseModel& noise)
      : Base(noise, key),
        world_point_(observation.world_point),
        measurement_(observation.measurement) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& pose,
                              gtsam::OptionalMatrixType jacobian = OptionalNone) const override {
    gtsam::Matrix point_jacobian;
    const gtsam::Point3 projected =
        pose.transformFrom(world_point_, jacobian == nullptr ? nullptr : &point_jacobian);
    if (jacobian != nullptr) *jacobian = point_jacobian;
    return projected - measurement_;
  }

 private:
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;
  gtsam::Point3 world_point_;
  gtsam::Point3 measurement_;
};

struct Result {
  double initial_cost;
  double final_cost;
  int iterations;
};

Result Solve(const std::vector<tinyopt::benchmark::robust_pose::Observation>& observations,
             const tinyopt::benchmark::robust_pose::Pose& initial_pose, bool robust) {
  gtsam::NonlinearFactorGraph graph;
  const auto gaussian = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);
  gtsam::SharedNoiseModel noise = gaussian;
  if (robust)
    noise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(0.3),
                                              gaussian);
  for (const auto& observation : observations)
    graph.emplace_shared<PointResidual>(PoseKey(), observation, noise);

  gtsam::Values initial;
  initial.insert(PoseKey(), ToPose(initial_pose));
  gtsam::LevenbergMarquardtParams options = gtsam::LevenbergMarquardtParams::CeresDefaults();
  options.maxIterations = 100;
  options.relativeErrorTol = 1e-6;
  options.absoluteErrorTol = 0;
  options.errorTol = 1e-12;
  options.lambdaInitial = 1e-4;
  options.minModelFidelity = 1e-12;
  options.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
  options.linearSolverType = gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;
  const double initial_cost = graph.error(initial);
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, options);
  const auto result = optimizer.optimize();
  return {initial_cost, graph.error(result), static_cast<int>(optimizer.iterations())};
}

void BenchmarkPose(int observation_count, bool robust) {
  using namespace tinyopt::benchmark::robust_pose;
  const Pose ground_truth = GroundTruth();
  const int outliers = observation_count / 4;
  const auto observations =
      MakeObservations(ground_truth, observation_count - outliers, outliers, 0.01, 50.0);
  const Result verification = Solve(observations, InitialPose(ground_truth), robust);
  REQUIRE(verification.iterations < 100);
  REQUIRE(verification.final_cost < verification.initial_cost);
  tinyopt::benchmark::PrintIterations(
      "Robust pose", std::to_string(observation_count) + "o " + (robust ? "Huber" : "L2"),
      "gtsam", verification.iterations, verification.iterations < 100);
  BENCHMARK(std::string(std::to_string(observation_count) + " obs " + (robust ? "Huber" : "L2"))) {
    return Solve(observations, InitialPose(ground_truth), robust).final_cost;
  };
}

}  // namespace

TEST_CASE("PoseOptimization", "[benchmark][robust][pose][gtsam]") {
  for (const int count : {20, 50, 100}) {
    BenchmarkPose(count, false);
    BenchmarkPose(count, true);
  }
}
