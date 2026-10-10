// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <string>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <ceres/ceres.h>

#include "sparse_problem.h"
#include "iterations.h"

namespace {

struct UnaryResidual : ceres::SizedCostFunction<1, 1> {
  explicit UnaryResidual(double target) : target(target) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    residuals[0] = parameters[0][0] - target;
    if (jacobians != nullptr && jacobians[0] != nullptr) jacobians[0][0] = 1;
    return true;
  }

  double target;
};

struct PairResidual : ceres::SizedCostFunction<1, 1, 1> {
  explicit PairResidual(double target_difference) : target_difference(target_difference) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    residuals[0] = 0.1 * ((parameters[1][0] - parameters[0][0]) - target_difference);
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) jacobians[0][0] = -0.1;
      if (jacobians[1] != nullptr) jacobians[1][0] = 0.1;
    }
    return true;
  }

  double target_difference;
};

struct Result {
  ceres::Solver::Summary summary;
  std::vector<double> parameters;
};

Result Solve(int dimensions) {
  Result result;
  result.parameters.resize(dimensions);
  ceres::Problem problem;
  for (int index = 0; index < dimensions; ++index) {
    result.parameters[index] = tinyopt::benchmark::sparse_problem::Initial(index);
    problem.AddParameterBlock(&result.parameters[index], 1);
    problem.AddResidualBlock(new UnaryResidual(tinyopt::benchmark::sparse_problem::Target(index)),
                             nullptr, &result.parameters[index]);
    if (index > 0) {
      problem.AddResidualBlock(
          new PairResidual(tinyopt::benchmark::sparse_problem::DifferenceTarget(index)), nullptr,
          &result.parameters[index - 1], &result.parameters[index]);
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 100;
  options.max_num_consecutive_invalid_steps = 3;
  options.num_threads = 1;
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-9;
  options.parameter_tolerance = 1e-8;
  options.min_relative_decrease = 1e-12;
  options.logging_type = ceres::SILENT;
  ceres::Solve(options, &problem, &result.summary);
  return result;
}

}  // namespace

TEST_CASE("Sparse", "[benchmark][sparse][ceres]") {
  const int dimensions = GENERATE(10, 100, 1000);
  CAPTURE(dimensions);
  const Result verification = Solve(dimensions);
  REQUIRE(verification.summary.IsSolutionUsable());
  REQUIRE(verification.summary.termination_type == ceres::CONVERGENCE);
  REQUIRE(verification.parameters.back() ==
          Catch::Approx(tinyopt::benchmark::sparse_problem::Target(dimensions - 1)).margin(1e-6));
  for (int index = 0; index < dimensions; ++index)
    REQUIRE(verification.parameters[index] ==
            Catch::Approx(tinyopt::benchmark::sparse_problem::Target(index)).margin(1e-6));
  tinyopt::benchmark::PrintIterations(
      "Sparse", std::to_string(dimensions) + "d", "ceres",
      static_cast<int>(verification.summary.iterations.size()) - 1,
      verification.summary.termination_type == ceres::CONVERGENCE);
  BENCHMARK(std::to_string(dimensions) + "D sparse chain") {
    return Solve(dimensions).summary.final_cost;
  };
}
