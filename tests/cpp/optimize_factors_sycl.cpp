// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimize_factors.h>
#include <tinyopt/optimize_factors_sycl.h>

using Catch::Approx;
using namespace tinyopt;

namespace {

struct QuadraticFactor {
  std::array<double, 2> target{1.0, -2.0};

  SyclFactorContribution<double, 2> operator()(
      Index index, const std::array<double, 2>& value) const {
    const std::size_t coordinate = static_cast<std::size_t>(index % 2);
    const double residual = value[coordinate] - target[coordinate];
    SyclFactorContribution<double, 2> contribution;
    contribution.cost = 0.5 * residual * residual;
    contribution.num_residuals = 1;
    contribution.gradient[coordinate] = residual;
    contribution.hessian[coordinate * 2 + coordinate] = 1.0;
    return contribution;
  }
};

void CheckFactorOptimization(sycl::queue& queue) {
  constexpr Index kFactorCount = 128;
  const QuadraticFactor factor;
  const Vec2 target(factor.target[0], factor.target[1]);
  const Vec2 initial_value(0.0, 0.0);
  const auto objective = [&](const Vec2& value, auto& gradient) {
    double cost = 0.0;
    for (Index i = 0; i < kFactorCount; ++i) {
      const auto contribution = factor(i, {value[0], value[1]});
      cost += contribution.cost;
      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        for (Index coordinate = 0; coordinate < 2; ++coordinate)
          gradient[coordinate] += contribution.gradient[static_cast<std::size_t>(coordinate)];
      }
    }
    return cost;
  };
  REQUIRE(diff::CheckGradient(initial_value, objective, 1e-5, diff::Method::kCentral, false));

  Options options;
  options.log.enable = false;
  options.exp.measure_time = false;
  options.exp.save_history = false;

  Vec2 cpu_value = initial_value;
  auto cpu_factor = [&](Index index, const Vec2& value, Vec2& gradient, Mat2& hessian) {
    const auto contribution = factor(index, {value[0], value[1]});
    for (Index row = 0; row < 2; ++row) {
      gradient[row] += contribution.gradient[static_cast<std::size_t>(row)];
      for (Index col = 0; col < 2; ++col) {
        hessian(row, col) += contribution.hessian[static_cast<std::size_t>(row * 2 + col)];
      }
    }
    return Cost(contribution.cost, contribution.num_residuals);
  };
  const Summary cpu_result = OptimizeFactors(cpu_value, kFactorCount, cpu_factor, options);
  REQUIRE(cpu_result.Succeeded());
  REQUIRE(cpu_result.Converged());

  Vec2 sycl_value = initial_value;
  const Summary sycl_result = OptimizeFactors(queue, sycl_value, kFactorCount, factor, options);
  REQUIRE(sycl_result.Succeeded());
  REQUIRE(sycl_result.Converged());
  REQUIRE((sycl_value - target).norm() < 1e-8);
  REQUIRE((sycl_value - cpu_value).norm() < 1e-10);
  REQUIRE(sycl_result.final_cost.cost == Approx(cpu_result.final_cost.cost).margin(1e-10));
  REQUIRE(sycl_result.final_cost.cost < 1e-12);

  REQUIRE_THROWS_AS(OptimizeFactors(queue, sycl_value, 0, factor, options), std::invalid_argument);
}

}  // namespace

#if !defined(TINYOPT_SYCL_TEST_GPU_ONLY)
TEST_CASE("SYCL OptimizeFactors computes and reduces CPU factors") {
  sycl::queue queue{sycl::cpu_selector_v};
  REQUIRE(queue.get_device().is_cpu());
  CheckFactorOptimization(queue);
}
#endif

#if defined(TINYOPT_SYCL_TEST_GPU)
TEST_CASE("SYCL OptimizeFactors computes and reduces GPU factors") {
  const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (devices.empty()) SKIP("No SYCL GPU device is available");
  sycl::queue queue{devices.front()};
  REQUIRE(queue.get_device().is_gpu());
  CheckFactorOptimization(queue);
}
#endif
