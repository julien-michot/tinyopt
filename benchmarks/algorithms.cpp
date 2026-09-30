// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/cg.h>
#include <tinyopt/optimizers/dl.h>

using namespace tinyopt;

#if defined(TINYOPT_ENABLE_CONJUGATE_GRADIENT)
TEST_CASE("Conjugate Gradient quadratic", "[benchmark][cg]") {
  Options options(Options::Solver::ConjugateGradient);
  options.log.enable = false;
  BENCHMARK("1D quadratic") {
    float x = 8.0f;
    return Optimize(x, [](const auto &value) {
      const auto residual = value - 2.0f;
      return residual * residual;
    }, options);
  };
}
#endif

#if defined(TINYOPT_ENABLE_DOGLEG)
TEST_CASE("DogLeg Rosenbrock", "[benchmark][dogleg]") {
  Options options(Options::Solver::DogLeg);
  options.log.enable = false;
  options.stop.max_iters = 200;
  const auto residuals = [](const auto &value) {
    using Scalar = typename std::decay_t<decltype(value)>::Scalar;
    Eigen::Matrix<Scalar, 2, 1> residual;
    residual[0] = Scalar(10) * (value[1] - value[0] * value[0]);
    residual[1] = Scalar(1) - value[0];
    return residual;
  };
  BENCHMARK("Rosenbrock 2D") {
    Vec2 x(-1.2, 1.0);
    return Optimize(x, residuals, options);
  };
}
#endif