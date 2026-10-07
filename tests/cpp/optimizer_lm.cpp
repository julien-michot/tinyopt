// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimizers/lm.h>
#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

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

TEST_CASE("tinyopt_lm_optimizer_jacobi_scaling") {
  Eigen::Vector2d parameters(1.0, -2.0);
  const auto loss = [](const auto &value, auto &gradient, auto &hessian) {
    const Eigen::Vector2d residual(1000.0 * value[0] - 1.0, value[1] - 2.0);
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.setZero();
      hessian.setZero();
      gradient[0] = 1000.0 * residual[0];
      gradient[1] = residual[1];
      hessian.coeffRef(0, 0) = 1e6;
      hessian.coeffRef(1, 1) = 1.0;
    }
    return Cost(0.5 * residual.squaredNorm(), 2);
  };
  Options options;
  options.lm.jacobi_scaling = true;

  const auto summary = lm::Optimizer<SparseMat>(options)(parameters, loss);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE(parameters[0] == Approx(1e-3).margin(1e-7));
  REQUIRE(parameters[1] == Approx(2.0).margin(1e-7));
}
