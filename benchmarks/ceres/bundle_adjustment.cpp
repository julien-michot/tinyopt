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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bundle_adjustment.h"
#include "bundle_adjustment_ceres.h"
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

Result Optimize(const Problem& problem) {
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif

  auto cameras = problem.initial_cameras;
  auto points = problem.initial_points;
  ceres::Problem ceres_problem;
  for (int camera = 0; camera < problem.CameraCount(); ++camera)
    ceres_problem.AddParameterBlock(cameras[camera].data(), Camera::Ambient, new SE3Manifold());
  for (int point = 0; point < problem.PointCount(); ++point)
    ceres_problem.AddParameterBlock(points[point].data(), 3);
  ceres_problem.SetParameterBlockConstant(cameras[0].data());
  ceres_problem.SetParameterBlockConstant(points[0].data());

  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  for (int point = 0; point < problem.PointCount(); ++point)
    ordering->AddElementToGroup(points[point].data(), 0);
  for (int camera = 0; camera < problem.CameraCount(); ++camera)
    ordering->AddElementToGroup(cameras[camera].data(), 1);
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
  options.function_tolerance = RelativeCostDecreaseTol;
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
  const auto dimensions =
      GENERATE(std::pair{2, 200}, std::pair{4, 600}, std::pair{10, 3000}, std::pair{100, 50000});
  const Problem problem = MakeProblem(dimensions.first, dimensions.second);
  const double reference_initial_cost = ReferenceReprojectionCost(problem);
  const Result verification = Optimize(problem);
  REQUIRE(verification.usable);
  REQUIRE(verification.iterations > 0);
  REQUIRE(verification.initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost < verification.initial_cost * 0.1);
  REQUIRE(verification.final_cost < MaxFinalCost(problem));
  const std::string label = ProblemLabel(problem);
  tinyopt::benchmark::PrintIterations("Bundle adjustment", label, "ceres", verification.iterations,
                                      verification.converged);

  BENCHMARK(std::string(label)) { return Optimize(problem).final_cost; };
}