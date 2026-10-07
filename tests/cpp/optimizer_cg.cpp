// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/cg.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_conjugate_gradient_optimizer_quadratic") {
  float x = 8.0f;
  Options options;
  options.cg.step_size = 0.25f;
  options.log.enable = false;
  options.stop.max_iters = 100;

  const auto objective = [](const auto &value, auto &gradient) {
    const float residual = value - 2.0f;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient(0) = 2.0f * residual;
    return residual * residual;
  };
  REQUIRE(diff::CheckGradient(x, objective));

  const auto sum = cg::Optimizer<Vec1f>(options)(x, objective);

  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x == Approx(2.0f).margin(1e-4f));
}

// Regression: the solver used `Grad_t::Zero()`, which is invalid for dynamic-size parameters.
TEST_CASE("tinyopt_conjugate_gradient_optimize_dynamic_parameters") {
  VecX x(2);
  x << 8.0, -8.0;
  const Vec2 target(2.0, -3.0);
  Options options(Options::Solver::ConjugateGradient);
  options.cg.step_size = 0.25f;
  options.log.enable = false;
  options.stop.max_iters = 100;

  const auto sum = Optimize(
      x,
      [&](const VecX &value, auto &gradient) {
        const Vec2 d = value - target;
        if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = 2.0 * d;
        return d.squaredNorm();
      },
      options);

  REQUIRE(sum.Succeeded());
  REQUIRE((x - target).norm() < 1e-3);
}

TEST_CASE("tinyopt_conjugate_gradient_optimize_dispatch") {
  Vec2 x(8.0f, -8.0f);
  const Vec2 target(2.0f, -3.0f);
  Options options(Options::Solver::ConjugateGradient);
  options.cg.step_size = 0.25f;
  options.log.enable = false;
  options.stop.max_iters = 100;

  const auto sum = Optimize(
      x,
      [&](const auto &value) {
        const auto dx = value[0] - target[0];
        const auto dy = value[1] - target[1];
        return dx * dx + dy * dy;
      },
      options);

  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE((x - target).norm() < 1e-4f);
}
