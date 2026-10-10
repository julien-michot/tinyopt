// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <utility>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

#include "bundle_adjustment.h"
#include "iterations.h"

using namespace tinyopt::benchmark::bundle_adjustment;

namespace {

gtsam::Key CameraKey(int index) { return gtsam::Symbol('c', index); }
gtsam::Key PointKey(int index) { return gtsam::Symbol('p', index); }

gtsam::Pose3 ToPose(const Camera& camera) {
  Eigen::Matrix3d world_to_camera;
  const Eigen::Vector3d angles = camera.head<3>();
  for (int column = 0; column < 3; ++column) {
    Eigen::Vector3d basis = Eigen::Vector3d::Zero();
    basis[column] = 1.0;
    world_to_camera.col(column) = RotateEuler(angles, basis);
  }
  const Eigen::Matrix3d camera_to_world = world_to_camera.transpose();
  const Eigen::Vector3d center = -camera_to_world * camera.tail<3>();
  return {gtsam::Rot3(camera_to_world), center};
}

gtsam::NonlinearFactorGraph MakeGraph(const Problem& problem) {
  using CameraFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;
  gtsam::NonlinearFactorGraph graph;
  const auto calibration =
      std::make_shared<gtsam::Cal3_S2>(FocalX, FocalY, 0.0, PrincipalX, PrincipalY);
  const auto measurement_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);

  for (const auto& observation : problem.observations) {
    graph.emplace_shared<CameraFactor>(observation.measurement, measurement_noise,
                                       CameraKey(observation.camera), PointKey(observation.point),
                                       calibration);
  }
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      CameraKey(0), ToPose(problem.initial_cameras[0]), gtsam::noiseModel::Constrained::All(6));
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(PointKey(0), problem.initial_points[0],
                                                          gtsam::noiseModel::Constrained::All(3));
  return graph;
}

gtsam::Values MakeInitialValues(const Problem& problem) {
  gtsam::Values initial;
  for (int camera = 0; camera < problem.CameraCount(); ++camera)
    initial.insert(CameraKey(camera), ToPose(problem.initial_cameras[camera]));
  for (int point = 0; point < problem.PointCount(); ++point)
    initial.insert(PointKey(point), problem.initial_points[point]);
  return initial;
}

struct Result {
  double initial_cost;
  double final_cost;
  int iterations;
};

gtsam::LevenbergMarquardtParams MakeOptions(const Problem& problem) {
  gtsam::LevenbergMarquardtParams options = gtsam::LevenbergMarquardtParams::CeresDefaults();
  options.maxIterations = 100;
  options.relativeErrorTol = 1e-6;
  options.absoluteErrorTol = 0;
  options.errorTol = 1e-12;
  options.lambdaInitial = 1e-4;
  options.minModelFidelity = 1e-12;
  options.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
  options.linearSolverType = gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;

  gtsam::Ordering ordering;
  for (int point = 0; point < problem.PointCount(); ++point) ordering.push_back(PointKey(point));
  for (int camera = 0; camera < problem.CameraCount(); ++camera)
    ordering.push_back(CameraKey(camera));
  options.setOrdering(ordering);
  return options;
}

Result Optimize(const Problem& problem) {
  const auto graph = MakeGraph(problem);
  const auto initial = MakeInitialValues(problem);
  const double initial_cost = graph.error(initial);
  gtsam::LevenbergMarquardtParams options = MakeOptions(problem);
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, options);
  const auto result = optimizer.optimize();
  const int iterations = static_cast<int>(optimizer.iterations());
  return {initial_cost, graph.error(result), iterations};
}

}  // namespace

TEST_CASE("BA", "[benchmark][bundle-adjustment][gtsam]") {
  const auto dimensions = GENERATE(std::pair{5, 50}, std::pair{20, 200}, std::pair{50, 500});
  const Problem problem = MakeProblem(dimensions.first, dimensions.second);
  const double reference_initial_cost = ReferenceReprojectionCost(problem);
  const Result verification = Optimize(problem);
  REQUIRE(verification.iterations > 0);
  REQUIRE(verification.iterations < 100);
  REQUIRE(verification.initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost < verification.initial_cost * 1e-4);
  REQUIRE(verification.final_cost < 1e-5);
  tinyopt::benchmark::PrintIterations("Bundle adjustment", ProblemLabel(problem), "gtsam",
                                      verification.iterations, verification.iterations < 100);

  const std::string label = ProblemLabel(problem);
  BENCHMARK(std::string(label)) { return Optimize(problem).final_cost; };
}