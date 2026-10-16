// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimizers/gd.h>

using namespace tinyopt;

TEST_CASE("tinyopt_gradient_descent_optimizer_quadratic") {
  float x = 0.0f;
  Options options;
  options.gd.lr = 0.1f;
  options.log.enable = false;
  options.stop.max_iters = 200;
  options.stop.min_error = 0;
  options.stop.min_rerr_dec = 0;

  const auto objective = [](const auto &value, auto &gradient) {
    const float residual = value - 1.0f;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient(0) = 2.0f * residual;
    return residual * residual;
  };
  REQUIRE(diff::CheckGradient(x, objective));

  const auto sum = gd::Optimizer<Vec1f>(options)(x, objective);

  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x == Catch::Approx(1.0f).margin(1e-4f));
}

TEST_CASE("tinyopt_gradient_descent_optimizer_stop_criteria_grad_norms") {
  Vec2f x(2.0f, 3.0f);
  const auto objective = [](const auto &val, auto &grad) {
    const float f0 = val(0);
    const float f1 = 2.0f * val(1);
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad(0) = 2.0f * f0;
      grad(1) = 8.0f * val(1);
    }
    return f0 * f0 + f1 * f1;
  };

  SECTION("stop on min_grad_max") {
    Options options;
    options.gd.lr = 0.05f;
    options.log.enable = false;
    options.stop.max_iters = 500;
    options.stop.min_error = 0;
    options.stop.min_rerr_dec = 0;
    options.stop.min_step_norm2 = 0;
    options.stop.min_grad_norm2 = 0;
    options.stop.min_grad_max = 1e-2f;

    gd::Optimizer<Vec2f> opt(options);
    REQUIRE_FALSE(opt.hasReduced());
    REQUIRE(opt.numReduced() == 0);

    Vec2f x_copy = x;
    const auto sum = opt(x_copy, objective);

    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
    REQUIRE(sum.stop_reason == StopReason::kMinGradNorm);
    REQUIRE(opt.GradientMaxNorm() <= 1e-2f);
    REQUIRE(opt.GradientSquaredNorm() == Catch::Approx(opt.Gradient().squaredNorm()));
    REQUIRE_FALSE(opt.hasReduced());
    REQUIRE(opt.numReduced() == 0);
  }

  SECTION("stop on min_grad_norm2") {
    Options options;
    options.gd.lr = 0.05f;
    options.log.enable = false;
    options.stop.max_iters = 500;
    options.stop.min_error = 0;
    options.stop.min_rerr_dec = 0;
    options.stop.min_step_norm2 = 0;
    options.stop.min_grad_norm2 = 1e-4f;
    options.stop.min_grad_max = 0;

    gd::Optimizer<Vec2f> opt(options);
    Vec2f x_copy = x;
    const auto sum = opt(x_copy, objective);

    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
    REQUIRE(sum.stop_reason == StopReason::kMinGradNorm);
    REQUIRE(opt.GradientSquaredNorm() <= 1e-4f);
    REQUIRE(opt.GradientSquaredNorm() == Catch::Approx(opt.Gradient().squaredNorm()));
  }
}