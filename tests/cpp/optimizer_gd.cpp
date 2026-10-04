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