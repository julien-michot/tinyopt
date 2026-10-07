// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimizers/dl.h>

using namespace tinyopt;

TEST_CASE("tinyopt_solver_dogleg_boundary_step") {
  Options options;
  options.solver_type = Options::Solver::DogLeg;
  options.dl.radius_init = 1.0f;
  dl::Optimizer<Mat2> optimizer(options);
  const Vec2 x = Vec2::Zero();
  const Vec2 gradient_value(3.0, 4.0);
  const auto accumulation = [&](const auto &, auto &gradient, auto &hessian) {
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient = gradient_value;
      hessian.setIdentity();
    }
    return Cost(100.0, 2);
  };

  REQUIRE(optimizer.Build(x, accumulation));
  const auto maybe_step = optimizer.SolveDogLeg();
  REQUIRE(maybe_step.has_value());
  REQUIRE(maybe_step->norm() == Catch::Approx(options.dl.radius_init).margin(1e-6));
  REQUIRE(maybe_step->dot(gradient_value) < 0.0);
}