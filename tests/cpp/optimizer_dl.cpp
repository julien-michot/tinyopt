// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/dl.h>

using namespace tinyopt;

TEST_CASE("tinyopt_dogleg_optimizer_rosenbrock") {
  Vec2 x(-1.2, 1.0);
  const auto residuals = [](const auto &value) {
    using Scalar = typename std::decay_t<decltype(value)>::Scalar;
    Eigen::Matrix<Scalar, 2, 1> residual;
    residual[0] = Scalar(10) * (value[1] - value[0] * value[0]);
    residual[1] = Scalar(1) - value[0];
    return residual;
  };

  const auto accumulation = [](const auto &value, auto &gradient, auto &hessian) {
    using Scalar = typename std::decay_t<decltype(value)>::Scalar;
    Eigen::Matrix<Scalar, 2, 1> residual;
    residual[0] = Scalar(10) * (value[1] - value[0] * value[0]);
    residual[1] = Scalar(1) - value[0];
    Eigen::Matrix<Scalar, 2, 2> jacobian;
    jacobian << -Scalar(20) * value[0], Scalar(10), Scalar(-1), Scalar(0);
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.noalias() = jacobian.transpose() * residual;
      hessian.noalias() = jacobian.transpose() * jacobian;
    }
    return residual;
  };

  REQUIRE(diff::CheckResidualsGradient(x, accumulation));
  Options options;
  options.dl.radius_init = 1.0f;
  options.dl.radius_max = 100.0f;
  options.log.enable = false;
  options.stop.max_iters = 500;
  options.stop.max_consec_failures = 20;

  const auto sum = dl::Optimizer<Mat2>(options)(x, residuals);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x[0] == Catch::Approx(1.0).margin(1e-4));
  REQUIRE(x[1] == Catch::Approx(1.0).margin(1e-4));
}

TEST_CASE("tinyopt_dogleg_optimize_dispatch") {
  Vec2 x(-1.2, 1.0);
  const auto residuals = [](const auto &value) {
    using Scalar = typename std::decay_t<decltype(value)>::Scalar;
    Eigen::Matrix<Scalar, 2, 1> result;
    result[0] = Scalar(10) * (value[1] - value[0] * value[0]);
    result[1] = Scalar(1) - value[0];
    return result;
  };
  Options options(Options::Solver::DogLeg);
  options.log.enable = false;
  options.stop.max_iters = 500;
  options.stop.max_consec_failures = 20;

  const auto sum = Optimize(x, residuals, options);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x[0] == Catch::Approx(1.0).margin(1e-4));
  REQUIRE(x[1] == Catch::Approx(1.0).margin(1e-4));
}
