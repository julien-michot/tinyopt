// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimizers/cg.h>

using namespace tinyopt;

TEST_CASE("tinyopt_solver_cg_initial_direction") {
  Options options;
  options.cg.step_size = 0.1f;
  options.solver_type = Options::Solver::ConjugateGradient;
  cg::Optimizer<Vec2> optimizer(options);
  const Vec2 x = Vec2::Zero();
  const Vec2 target(4, 5);
  const auto accumulation = [&](const auto &, auto &gradient) {
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = -target;
    return Cost(1.0, 2);
  };

  REQUIRE(optimizer.Build(x, accumulation));
  const auto maybe_dx = optimizer.Solve();
  REQUIRE(maybe_dx.has_value());
  REQUIRE((*maybe_dx - options.cg.step_size * target).norm() < 1e-5);
}