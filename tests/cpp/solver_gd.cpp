// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/gd.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_solver_gd_step") {
  Options options;
  options.gd.lr = 0.1f;
  gd::Optimizer<Vec2> optimizer(options);
  const Vec2 x = Vec2::Zero();
  const Vec2 target(4, 5);
  const auto accumulation = [&](const auto &, auto &gradient) {
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = -target;
    return Cost(1.0, 2);
  };

  REQUIRE(optimizer.Build(x, accumulation));
  const auto maybe_dx = optimizer.Solve();
  REQUIRE(maybe_dx.has_value());
  REQUIRE((*maybe_dx - options.gd.lr * target).norm() < 1e-5);
}

TEST_CASE("tinyopt_solver_gd_templated_gradient_accumulator") {
  Vec2 x = Vec2::Zero();
  const Vec2 target(4, 5);
  Options options(Options::Solver::GradientDescent);
  options.gd.lr = 0.1f;
  options.stop.max_iters = 100;
  options.log.enable = false;

  const auto accumulation = [&](const auto &value, auto &gradient) {
    const auto residual = value - target;
    gradient = 2 * residual;
    return residual.squaredNorm();
  };

  const auto summary = Optimize(x, accumulation, options);

  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE((x - target).norm() < 1e-5);
}