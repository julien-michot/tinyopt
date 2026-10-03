// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <memory>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ceres/ceres.h>

#include "bundle_adjustment.h"

using namespace tinyopt::benchmark::bundle_adjustment;

namespace {

struct Result {
  double initial_cost;
  double final_cost;
  bool usable;
  int successful_steps;
};

struct ReprojectionCost {
  explicit ReprojectionCost(const ImagePoint& measurement) : measurement_(measurement) {}

  template <typename Scalar>
  bool operator()(const Scalar* camera, const Scalar* point, Scalar* residuals) const {
    Eigen::Matrix<Scalar, 6, 1> pose;
    Eigen::Matrix<Scalar, 3, 1> landmark;
    for (int index = 0; index < 6; ++index) pose[index] = camera[index];
    for (int index = 0; index < 3; ++index) landmark[index] = point[index];
    const Eigen::Matrix<Scalar, 2, 1> residual =
        Project(pose, landmark) - measurement_.cast<Scalar>();
    residuals[0] = residual[0];
    residuals[1] = residual[1];
    return true;
  }

  ImagePoint measurement_;
};

Result Optimize(const Problem& problem) {
  auto cameras = problem.initial_cameras;
  auto points = problem.initial_points;
  ceres::Problem ceres_problem;
  for (int camera = 0; camera < CameraCount; ++camera)
    ceres_problem.AddParameterBlock(cameras[camera].data(), 6);
  for (int point = 0; point < PointCount; ++point)
    ceres_problem.AddParameterBlock(points[point].data(), 3);
  ceres_problem.SetParameterBlockConstant(cameras[0].data());
  ceres_problem.SetParameterBlockConstant(points[0].data());

  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  for (int camera = 1; camera < CameraCount; ++camera)
    ordering->AddElementToGroup(cameras[camera].data(), 0);
  for (int point = 1; point < PointCount; ++point)
    ordering->AddElementToGroup(points[point].data(), 1);
  for (const auto& observation : problem.observations) {
    ceres_problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 6, 3>(
                                       new ReprojectionCost(observation.measurement)),
                                   nullptr, cameras[observation.camera].data(),
                                   points[observation.point].data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.initial_trust_region_radius = 1e4;
  options.max_num_iterations = 10;
  options.max_num_consecutive_invalid_steps = 3;
  options.num_threads = 1;
  options.function_tolerance = 1e-12;
  options.gradient_tolerance = 1e-9;
  options.parameter_tolerance = 1e-8;
  options.min_relative_decrease = 1e-12;
  options.logging_type = ceres::SILENT;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &ceres_problem, &summary);
  return {summary.initial_cost, summary.final_cost, summary.IsSolutionUsable(),
          summary.num_successful_steps};
}

}  // namespace

TEST_CASE("BA", "[benchmark][bundle-adjustment][ceres]") {
  const Problem problem = MakeProblem();
  const double reference_initial_cost = ReferenceReprojectionCost(problem);
  const Result verification = Optimize(problem);
  REQUIRE(verification.usable);
  REQUIRE(verification.successful_steps > 0);
  REQUIRE(verification.initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost < verification.initial_cost * 1e-4);
  REQUIRE(verification.final_cost < 1e-6);

  BENCHMARK("5 cams, 50 pts") { return Optimize(problem).final_cost; };
}