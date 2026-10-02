// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include <Eigen/Eigen>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/optimizer.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::solvers;

TEST_CASE("tinyopt_variadic_optimize_in_out_parameters") {
  double x = 0.0;
  double y = 0.0;

  auto cost = [](const auto &a, const auto &b) {
    return (a - 3.0) * (a - 3.0) + (b + 1.0) * (b + 1.0);
  };

  auto out = Optimize(x, y, cost);
  REQUIRE(out.Succeeded());
  REQUIRE(std::isfinite(x));
  REQUIRE(std::isfinite(y));
  REQUIRE(out.final_cost.cost < 100.0);
  REQUIRE(x != Approx(0.0).margin(1e-8));
  REQUIRE(y != Approx(0.0).margin(1e-8));

  x = 0.0;
  y = 0.0;
  using Optimizer = Optimizer_<SolverLM<Mat2>>;
  Optimizer::Options options;
  options.max_iters = 200;
  options.max_consec_failures = 20;
  Optimizer optimizer(options);

  auto out2 = optimizer(x, y, cost);
  REQUIRE(out2.Succeeded());
  REQUIRE(std::isfinite(x));
  REQUIRE(std::isfinite(y));
  REQUIRE(out2.final_cost.cost < 100.0);
  REQUIRE(x != Approx(0.0).margin(1e-8));
  REQUIRE(y != Approx(0.0).margin(1e-8));
}
