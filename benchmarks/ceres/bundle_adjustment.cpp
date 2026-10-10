// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <utility>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <ceres/ceres.h>

#include "bundle_adjustment.h"
#include "iterations.h"

using namespace tinyopt::benchmark::bundle_adjustment;

namespace {

struct Result {
  double initial_cost;
  double final_cost;
  bool usable;
  int iterations;
  bool converged;
};

struct ReprojectionCost final : ceres::SizedCostFunction<2, 6, 3> {
  explicit ReprojectionCost(const ImagePoint& measurement) : measurement_(measurement) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const Eigen::Map<const Camera> camera(parameters[0]);
    const Eigen::Map<const Point> point(parameters[1]);
    Eigen::Matrix<double, 2, 6> camera_jacobian;
    Eigen::Matrix<double, 2, 3> point_jacobian;
    ImagePoint projected;
    ProjectionJacobians(camera, point, projected, camera_jacobian, point_jacobian);
    Eigen::Map<ImagePoint> output(residuals);
    output = projected - measurement_;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[0]);
      J = camera_jacobian;
    }
    if (jacobians != nullptr && jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
      J = point_jacobian;
    }
    return true;
  }

  ImagePoint measurement_;
};

Result Optimize(const Problem& problem) {
  auto cameras = problem.initial_cameras;
  auto points = problem.initial_points;
  ceres::Problem ceres_problem;
  for (int camera = 0; camera < problem.CameraCount(); ++camera)
    ceres_problem.AddParameterBlock(cameras[camera].data(), 6);
  for (int point = 0; point < problem.PointCount(); ++point)
    ceres_problem.AddParameterBlock(points[point].data(), 3);
  ceres_problem.SetParameterBlockConstant(cameras[0].data());
  ceres_problem.SetParameterBlockConstant(points[0].data());

  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  for (int camera = 1; camera < problem.CameraCount(); ++camera)
    ordering->AddElementToGroup(cameras[camera].data(), 0);
  for (int point = 1; point < problem.PointCount(); ++point)
    ordering->AddElementToGroup(points[point].data(), 1);
  for (const auto& observation : problem.observations) {
    ceres_problem.AddResidualBlock(new ReprojectionCost(observation.measurement), nullptr,
                                   cameras[observation.camera].data(),
                                   points[observation.point].data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.initial_trust_region_radius = 1e4;
  options.max_num_iterations = 100;
  options.max_num_consecutive_invalid_steps = 3;
  options.num_threads = 1;
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-9;
  options.parameter_tolerance = 1e-8;
  options.min_relative_decrease = 1e-12;
  options.logging_type = ceres::SILENT;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &ceres_problem, &summary);
  return {summary.initial_cost, summary.final_cost, summary.IsSolutionUsable(),
          static_cast<int>(summary.iterations.size()) - 1,
          summary.termination_type == ceres::CONVERGENCE};
}

}  // namespace

TEST_CASE("BA", "[benchmark][bundle-adjustment][ceres]") {
  const auto dimensions = GENERATE(std::pair{5, 50}, std::pair{20, 200}, std::pair{50, 500});
  const Problem problem = MakeProblem(dimensions.first, dimensions.second);
  const double reference_initial_cost = ReferenceReprojectionCost(problem);
  const Result verification = Optimize(problem);
  REQUIRE(verification.usable);
  REQUIRE(verification.iterations > 0);
  REQUIRE(verification.initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost < verification.initial_cost * 1e-4);
  REQUIRE(verification.final_cost < 1e-5);
  tinyopt::benchmark::PrintIterations("Bundle adjustment", ProblemLabel(problem), "ceres",
                                      verification.iterations, verification.converged);

  const std::string label = ProblemLabel(problem);
  BENCHMARK(std::string(label)) { return Optimize(problem).final_cost; };
}