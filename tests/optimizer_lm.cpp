// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/tinyopt.h>
#include <tinyopt/optimizers/lm.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::solvers;

TEST_CASE("tinyopt_lm_optimizer_manual_accumulation") {
  float x = 1.0f;
  const auto loss = [](const auto &value, auto &gradient, auto &hessian) {
    const float residual = value * value - 2.0f;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      const float jacobian = 2.0f * value;
      gradient(0) = jacobian * residual;
      hessian(0, 0) = jacobian * jacobian;
    }
    return std::abs(residual);
  };

  const auto sum = lm::Optimizer<Mat1f>{}(x, loss);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
}

TEST_CASE("tinyopt_lm_optimizer_autodiff") {
  float x = 1.0f;
  const auto loss = [](const auto &value) {
    using Scalar = std::decay_t<decltype(value)>;
    return value * value - Scalar(2.0);
  };

  const auto sum = lm::Optimizer<Vec1f>{}(x, loss);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
}