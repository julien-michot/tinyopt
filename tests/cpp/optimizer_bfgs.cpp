// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/bfgs.h>

using namespace tinyopt;

template <typename Optimizer>
void CheckQuasiNewtonQuadratic() {
  Vec2 x(8.0, -8.0);
  const Vec2 target(2.0, -3.0);
  Options options;
  options.log.enable = false;
  options.stop.max_iters = 300;
  options.stop.min_rerr_dec = 0;
  options.stop.min_error = 1e-14f;
  options.bfgs.step_size = 0.25f;

  const auto objective = [&](const auto &value) { return (value - target).squaredNorm(); };
  REQUIRE(diff::CheckGradient(x, [&](const auto &value, auto &gradient) {
    const auto residual = value - target;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = 2.0 * residual;
    return residual.squaredNorm();
  }));

  const auto sum = Optimizer(options)(x, objective);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE((x - target).norm() < 1e-5);
}

TEST_CASE("tinyopt_optimizer_bfgs_quadratic") {
  CheckQuasiNewtonQuadratic<bfgs::Optimizer<Vec2>>();
}

TEST_CASE("tinyopt_optimizer_lbfgs_quadratic") {
  CheckQuasiNewtonQuadratic<lbfgs::Optimizer<Vec2>>();
}

template <typename Optimizer>
void CheckRosenbrock(Options options) {
  Vec2 x(-1.2, 1.0);
  options.log.enable = false;
  options.stop.max_iters = 5000;
  options.stop.max_consec_failures = 20;
  options.stop.min_rerr_dec = 0;
  options.stop.min_error = 1e-12f;
  options.bfgs.step_size = 0.005f;

  const auto objective = [](const auto &value, auto &gradient) {
    const auto valley = value[1] - value[0] * value[0];
    const auto one_minus_x = 1.0 - value[0];
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient[0] = -400.0 * value[0] * valley - 2.0 * one_minus_x;
      gradient[1] = 200.0 * valley;
    }
    return 100.0 * valley * valley + one_minus_x * one_minus_x;
  };
  REQUIRE(diff::CheckGradient(x, objective));

  const auto sum = Optimizer(options)(x, objective);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x[0] == Catch::Approx(1.0).margin(1e-4));
  REQUIRE(x[1] == Catch::Approx(1.0).margin(1e-4));
}

TEST_CASE("tinyopt_optimizer_bfgs_rosenbrock") {
  CheckRosenbrock<bfgs::Optimizer<Vec2>>(Options{});
}

TEST_CASE("tinyopt_optimizer_lbfgs_scaled_quadratic") {
  Vec2 x(8.0, -8.0);
  const Vec2 target(2.0, -3.0);
  Options options;
  options.log.enable = false;
  options.stop.max_iters = 5000;
  options.stop.max_consec_failures = 20;
  options.stop.min_rerr_dec = 0;
  options.stop.min_error = 1e-12f;
  options.lbfgs.step_size = 0.01f;

  const auto objective = [&](const auto &value, auto &gradient) {
    const auto first = value[0] - target[0];
    const auto second = value[1] - target[1];
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient[0] = 2.0 * first;
      gradient[1] = 20.0 * second;
    }
    return first * first + 10.0 * second * second;
  };
  REQUIRE(diff::CheckGradient(x, objective));

  const auto sum = lbfgs::Optimizer<Vec2>(options)(x, objective);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE((x - target).norm() < 1e-5);
}

#if defined(TINYOPT_ENABLE_BFGS)
TEST_CASE("tinyopt_bfgs_optimize_dispatch") {
  Vec2 x(8.0, -8.0);
  const Vec2 target(2.0, -3.0);
  Options options(Options::Solver::BFGS);
  options.log.enable = false;
  options.stop.max_iters = 300;
  options.stop.min_rerr_dec = 0;
  options.stop.min_error = 1e-14f;
  options.bfgs.step_size = 0.25f;
  const auto sum =
      Optimize(x, [&](const auto &value) { return (value - target).squaredNorm(); }, options);

  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE((x - target).norm() < 1e-5);
}
#else
TEST_CASE("disabled_bfgs_is_not_in_optimize_dispatch") {
  double x = 0;
  Options options(Options::Solver::BFGS);
  REQUIRE_THROWS(Optimize(x, [](const auto &value) { return value * value; }, options));
}
#endif

#if defined(TINYOPT_ENABLE_LBFGS)
TEST_CASE("tinyopt_lbfgs_optimize_dispatch") {
  Vec2 x(8.0, -8.0);
  const Vec2 target(2.0, -3.0);
  Options options(Options::Solver::LBFGS);
  options.log.enable = false;
  options.stop.max_iters = 300;
  options.stop.min_rerr_dec = 0;
  options.stop.min_error = 1e-14f;
  options.lbfgs.step_size = 0.25f;
  const auto sum =
      Optimize(x, [&](const auto &value) { return (value - target).squaredNorm(); }, options);

  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE((x - target).norm() < 1e-5);
}
#else
TEST_CASE("disabled_lbfgs_is_not_in_optimize_dispatch") {
  double x = 0;
  Options options(Options::Solver::LBFGS);
  REQUIRE_THROWS(Optimize(x, [](const auto &value) { return value * value; }, options));
}
#endif