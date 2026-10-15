// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <utility>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tinyopt/tinyopt.h>

#include "bundle_adjustment.h"
#include "bundle_adjustment_tinyopt.h"
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::benchmark::bundle_adjustment;

TEST_CASE("BA block-sparse", "[benchmark][bundle-adjustment][sparse]") {
  const auto dimensions =
      GENERATE(std::pair{2, 200}, std::pair{4, 600}, std::pair{10, 3000}, std::pair{100, 50000});
  const Problem initial_problem = MakeProblem(dimensions.first, dimensions.second);
  Problem verification_problem = initial_problem;
  const BlockSparseLoss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.lm.jacobi_scaling = true;
  const double reference_initial_cost = ReferenceReprojectionCost(initial_problem);

  using BlockHessian = BlockSparseLoss::Hessian;
  lm::Optimizer<BlockHessian> verification_optimizer(options);
  const auto& verification = verification_optimizer(verification_problem, loss);
  std::nullptr_t null_gradient{};
  BlockHessian unused_hessian;
  const double initial_cost = loss(initial_problem, null_gradient, unused_hessian).cost;
  REQUIRE(verification.Succeeded());
  REQUIRE(verification.Converged());
  REQUIRE(initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost.cost < initial_cost * 1e-4);
  REQUIRE(verification.final_cost.cost < 1e-5);
  REQUIRE(verification_problem.initial_cameras[0] == initial_problem.initial_cameras[0]);
  REQUIRE(verification_problem.initial_points[0] == initial_problem.initial_points[0]);
  tinyopt::benchmark::PrintIterations("Bundle adjustment",
                                      ProblemLabel(initial_problem) + " block-sparse", "tinyopt",
                                      verification.num_iters, verification.Converged());

  const std::string label = ProblemLabel(initial_problem) + " block-sparse";
  BENCHMARK(std::string(label)) {
    Problem problem = initial_problem;
    lm::Optimizer<BlockHessian> optimizer(options);
    const auto& result = optimizer(problem, loss);
    return result.final_cost.cost;
  };
}

TEST_CASE("BA sparse", "[benchmark][bundle-adjustment][sparse]") {
  const auto dimensions =
      GENERATE(std::pair{2, 200}, std::pair{4, 600}, std::pair{10, 3000}, std::pair{100, 50000});
  const Problem initial_problem = MakeProblem(dimensions.first, dimensions.second);
  Problem verification_problem = initial_problem;
  const SparseLoss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.lm.jacobi_scaling = true;
  const double reference_initial_cost = ReferenceReprojectionCost(initial_problem);

  using Hessian = SparseLoss::Hessian;
  lm::Optimizer<Hessian> verification_optimizer(options);
  const auto& verification = verification_optimizer(verification_problem, loss);
  std::nullptr_t null_gradient{};
  Hessian unused_hessian;
  const double initial_cost = loss(initial_problem, null_gradient, unused_hessian).cost;
  REQUIRE(verification.Succeeded());
  REQUIRE(verification.Converged());
  REQUIRE(initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost.cost < initial_cost * 1e-4);
  REQUIRE(verification.final_cost.cost < 1e-5);
  REQUIRE(verification_problem.initial_cameras[0] == initial_problem.initial_cameras[0]);
  REQUIRE(verification_problem.initial_points[0] == initial_problem.initial_points[0]);
  tinyopt::benchmark::PrintIterations("Bundle adjustment", ProblemLabel(initial_problem), "tinyopt",
                                      verification.num_iters, verification.Converged());

  const std::string label = ProblemLabel(initial_problem);
  BENCHMARK(std::string(label)) {
    Problem problem = initial_problem;
    lm::Optimizer<Hessian> optimizer(options);
    const auto& result = optimizer(problem, loss);
    return result.final_cost.cost;
  };
}