// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;

using Catch::Approx;

void TestSimpleGradientDescent() {
  auto loss = [&](const auto &x, auto &grad) {
    double y = x - 42.0;
    double cost = 3 * y * y + std::pow(y, 4.0) - 2.0;
    // Manually update the gradient
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad(0) = 2 * 3 * y + 4 * std::pow(y, 3.0);
    }
    return cost;
  };

  double x = 40.1;
  REQUIRE(diff::CheckGradient(x, loss, 1e-3));
  Options options;  // These are common options
  options.solver_type = Options::Solver::GradientDescent;
  options.max_iters = 1000; // let's say it's not the fastest optimizer...
  options.min_error = 0;
  options.min_rerr_dec = 0;
  options.gd.lr = 0.01; // especially with this!
  const auto &out = Optimize(x, loss, options);
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(42.0).margin(1e-5));
}

TEST_CASE("tinyopt_unconstrained") { TestSimpleGradientDescent(); }