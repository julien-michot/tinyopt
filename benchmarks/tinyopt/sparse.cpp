// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tinyopt/tinyopt.h>

#include "options.h"
#include "sparse_problem.h"
#include "iterations.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;

namespace {

struct SparseChain {
  Cost operator()(const VecX& x, auto& gradient, SparseMat& hessian) const {
    const Index dimensions = x.size();
    double cost = 0;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.setZero();
      hessian.setZero();
    }

    for (Index index = 0; index < dimensions; ++index) {
      const double target = sparse_problem::Target(static_cast<int>(index));
      const double residual = x[index] - target;
      cost += residual * residual;
      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        gradient[index] += residual;
        hessian.coeffRef(index, index) += 1;
      }
    }

    for (Index index = 0; index + 1 < dimensions; ++index) {
      const double target = sparse_problem::DifferenceTarget(static_cast<int>(index + 1));
      const double residual = 0.1 * ((x[index + 1] - x[index]) - target);
      cost += residual * residual;
      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        gradient[index] -= 0.1 * residual;
        gradient[index + 1] += 0.1 * residual;
        hessian.coeffRef(index, index) += 0.01;
        hessian.coeffRef(index, index + 1) -= 0.01;
        hessian.coeffRef(index + 1, index) -= 0.01;
        hessian.coeffRef(index + 1, index + 1) += 0.01;
      }
    }
    return Cost(cost, static_cast<int>(2 * dimensions - 1));
  }
};

VecX SparseInitial(Index dimensions) {
  VecX initial(dimensions);
  for (Index index = 0; index < dimensions; ++index)
    initial[index] = sparse_problem::Initial(static_cast<int>(index));
  return initial;
}

VecX SparseTarget(Index dimensions) {
  VecX target(dimensions);
  for (Index index = 0; index < dimensions; ++index)
    target[index] = sparse_problem::Target(static_cast<int>(index));
  return target;
}

}  // namespace

TEST_CASE("Sparse", "[benchmark][sparse]") {
  const Index dimensions = GENERATE(10, 100, 1000);
  CAPTURE(dimensions);
  const SparseChain loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.hessian.H_is_full = true;

  VecX verification = SparseInitial(dimensions);
  lm::Optimizer<SparseMat> optimizer(options);
  const auto& output = optimizer(verification, loss);
  REQUIRE(output.Succeeded());
  REQUIRE(output.Converged());
  REQUIRE((verification - SparseTarget(dimensions)).norm() < 1e-5);
  PrintIterations("Sparse", std::to_string(dimensions) + "d", "tinyopt", output.num_iters,
                  output.Converged());

  BENCHMARK(std::to_string(dimensions) + "D sparse chain") {
    VecX x = SparseInitial(dimensions);
    optimizer.reset();
    const auto result = optimizer(x, loss);
    return result.final_cost.cost;
  };
}
