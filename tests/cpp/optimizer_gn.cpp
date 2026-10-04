// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimizers/gn.h>

using namespace tinyopt;

TEST_CASE("tinyopt_gauss_newton_optimizer_sqrt2") {
  float x = 1.0f;
  Options options;
  options.log.enable = false;
  const auto sum =
      gn::Optimizer<Mat1f>(options)(x, [](const auto &value) { return value * value - 2.0f; });

  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x == Catch::Approx(std::sqrt(2.0f)).margin(1e-5f));
}