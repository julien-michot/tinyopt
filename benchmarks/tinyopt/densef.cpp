// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <type_traits>

#include <tinyopt/tinyopt.h>

#include "dense_loss.h"
#include "dense_problems.h"
#include "iterations.h"
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;

namespace {

template <typename Vector>
void RunFloatBenchmark() {
  constexpr Index FixedDimensions = Vector::RowsAtCompileTime;
  const Index dimensions = FixedDimensions == Eigen::Dynamic ? GENERATE(1, 2, 3) : FixedDimensions;
  const std::string storage = FixedDimensions == Eigen::Dynamic ? "dynamic" : "static";
  const std::string label = std::to_string(dimensions) + "D " + storage + " float";
  auto initial = DenseMathInitial<float>(dimensions);
  Vector parameters = initial;
  const DenseMathLoss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.stop.min_error = 1e-6;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_step_norm2 = 1e-10;
  using Hessian = Eigen::Matrix<float, Vector::RowsAtCompileTime, Vector::RowsAtCompileTime>;
  lm::Optimizer<Hessian> verification_optimizer(options);
  const auto scalar_loss = [&loss](const auto& x, auto& gradient) {
    std::nullptr_t null_hessian{};
    return loss(x, gradient, null_hessian).cost;
  };
  REQUIRE(diff::CheckGradient(parameters, scalar_loss, 1e-2, diff::Method::kCentral, false));
  const auto result = verification_optimizer(parameters, loss);
  INFO(StopReasonDescription(result, options));
  REQUIRE(result.Succeeded());
  REQUIRE(result.Converged());
  const float expected = dimensions == 1 ? std::sqrt(2.0f) : 1.0f;
  if (dimensions == 1)
    REQUIRE(std::abs(parameters[0] - expected) < 1e-4f);
  else
    REQUIRE((parameters.array() - 1.0f).matrix().norm() < 1e-4f);
  tinyopt::benchmark::PrintIterations("Dense " + storage, std::to_string(dimensions) + "f",
                                      "tinyopt", result.num_iters, result.Converged());
  lm::Optimizer<Hessian> optimizer(options);
  BENCHMARK(std::string(label)) {
    Vector x = initial;
    optimizer.reset();
    if constexpr (std::is_same_v<Vector, Vec1f>) {
      float x_scalar = x[0];
      const auto output = optimizer(x_scalar, loss);
      x[0] = x_scalar;
      return output.final_cost.cost;
    } else {
      const auto output = optimizer(x, loss);
      return output.final_cost.cost;
    }
  };
}

}  // namespace

TEMPLATE_TEST_CASE("Dense", "[benchmark][dense][float]", Vec1f, Vec2f, Vec3f, VecXf) {
  RunFloatBenchmark<TestType>();
}
