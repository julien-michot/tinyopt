// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <type_traits>
#include <type_traits>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tinyopt/tinyopt.h>

#include "dense_problems.h"
#include "dense_loss.h"
#include "iterations.h"
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;

namespace {

template <typename Vector>
void CheckMathProblem(Index dimensions) {
  using Scalar = typename Vector::Scalar;
  auto initial_dynamic = DenseMathInitial<Scalar>(dimensions);
  Vector initial;
  initial = initial_dynamic;
  const DenseMathLoss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.stop.min_error = 1e-14;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_step_norm2 = 1e-16;
  using Hessian = Eigen::Matrix<Scalar, Vector::RowsAtCompileTime, Vector::RowsAtCompileTime>;
  lm::Optimizer<Hessian> optimizer(options);
  const auto scalar_loss = [&loss](const auto& x, auto& gradient) {
    std::nullptr_t null_hessian{};
    return loss(x, gradient, null_hessian).cost;
  };
  REQUIRE(diff::CheckGradient(initial, scalar_loss, 1e-4, diff::Method::kCentral, false));
  const auto result = optimizer(initial, loss);
  INFO(StopReasonDescription(result, options));
  REQUIRE(result.Succeeded());
  REQUIRE(result.Converged());
  const std::string storage =
      Vector::RowsAtCompileTime == Eigen::Dynamic ? "dynamic" : "static";
  const std::string type = std::is_same_v<Scalar, float> ? "f" : "d";
  tinyopt::benchmark::PrintIterations(
      "Dense " + storage, std::to_string(dimensions) + type, "tinyopt", result.num_iters,
      result.Converged());
  if (dimensions == 1) {
    REQUIRE(std::abs(initial[0] - std::sqrt(Scalar(2))) < Scalar(1e-5));
  } else {
    REQUIRE((initial.array() - Scalar(1)).matrix().norm() < Scalar(1e-5));
  }
}

template <typename Vector>
void RunMathBenchmark(Index dimensions) {
  using Scalar = typename Vector::Scalar;
  constexpr Index FixedDimensions = Vector::RowsAtCompileTime;
  const std::string storage = FixedDimensions == Eigen::Dynamic ? "dynamic" : "static";
  const std::string label = std::to_string(dimensions) + "D " + storage + " " +
                            (std::is_same_v<Scalar, float> ? "float" : "double");
  CheckMathProblem<Vector>(dimensions);
  const DenseMathLoss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.stop.min_error = 1e-14;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_step_norm2 = 1e-16;
  using Hessian = Eigen::Matrix<Scalar, Vector::RowsAtCompileTime, Vector::RowsAtCompileTime>;
  lm::Optimizer<Hessian> optimizer(options);
  BENCHMARK(std::string(label)) {
    Vector x = DenseMathInitial<Scalar>(dimensions);
    optimizer.reset();
    const auto result = optimizer(x, loss);
    return result.final_cost.cost;
  };
}

template <typename Vector>
void RunPriorBenchmark(Index dimensions) {
  using Scalar = typename Vector::Scalar;
  Vector initial = PriorInitial<Scalar>(dimensions);
  const Vector target = PriorTarget<Scalar>(dimensions);
  const auto loss = [&target](const auto& x, auto& gradient, auto& hessian) {
    const auto residual = x - target;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient = residual;
      hessian.setIdentity();
    }
    return Cost(0.5 * residual.squaredNorm(), static_cast<int>(x.size()));
  };
  Options options = CreateOptions();
  options.stop.max_iters = 20;
  options.stop.min_error = 1e-14;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_step_norm2 = 1e-16;
  lm::Optimizer<MatX> optimizer(options);
  const auto result = optimizer(initial, loss);
  INFO(StopReasonDescription(result, options));
  REQUIRE(result.Succeeded());
  REQUIRE(result.Converged());
  REQUIRE((initial - target).norm() < 1e-8);
  tinyopt::benchmark::PrintIterations("Dense dynamic", std::to_string(dimensions) + "dp",
                                      "tinyopt", result.num_iters, result.Converged());
  BENCHMARK(std::to_string(dimensions) + "D dynamic double prior") {
    Vector x = PriorInitial<Scalar>(dimensions);
    optimizer.reset();
    const auto output = optimizer(x, loss);
    return output.final_cost.cost;
  };
}

}  // namespace

TEMPLATE_TEST_CASE("Dense", "[benchmark][dense][double]", Vec1, Vec2, Vec3) {
  RunMathBenchmark<TestType>(TestType::RowsAtCompileTime);
}

TEMPLATE_TEST_CASE("Dense", "[benchmark][dense][double]", VecX) {
  for (const Index dimensions : {1, 2, 3}) RunMathBenchmark<TestType>(dimensions);
  for (const Index dimensions : {6, 12, 33, 50}) RunPriorBenchmark<TestType>(dimensions);
}
