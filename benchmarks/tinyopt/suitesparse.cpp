// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tinyopt/tinyopt.h>
#include "iterations.h"
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;

auto sparse_prior = [](const auto &x, auto &gradient, SparseMat &hessian) {
  const VecX residual = 10.0 * x.array() - 2.0;
  if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
    gradient = 10.0 * residual;
    hessian.setIdentity();
    hessian *= 100.0;
  }
  return Cost(residual.norm(), residual.size());
};

TEST_CASE("SuiteSparse", "[benchmark][dyn][sparse][suitesparse]") {
  const auto dims = GENERATE(10, 100, 1000);
  CAPTURE(dims);

  Options options = CreateOptions();
  options.linear_solver = LinearSolverMethod::SuiteSparse;
  options.stop.max_iters = 100;
  lm::Optimizer<SparseMat> optimizer(options);
  VecX verification = VecX::Random(dims);
  const auto result = optimizer(verification, sparse_prior);
  REQUIRE(result.Succeeded());
  REQUIRE(result.Converged());
  PrintIterations("Sparse", std::to_string(dims) + "d", "tinyopt", result.num_iters,
                  result.Converged());

  BENCHMARK(std::to_string(dims) + "x" + std::to_string(dims) + " SuiteSparse prior") {
    VecX x = VecX::Random(dims);
    optimizer.reset();
    return optimizer(x, sparse_prior).final_cost.cost;
  };
}