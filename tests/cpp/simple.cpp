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
#include "explicit_instantiation.h"

using namespace tinyopt;
using namespace tinyopt::nlls;

using Catch::Approx;

void TestSimpleLM() {
  auto loss = [&](const auto &x, auto &grad, auto &H) {
    double res = x - 2;
    // Manually update the H and gradient (J is 1 here)
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad(0) = res;
      H(0, 0) = 1;
    }
    return std::abs(res);  // Returns the error norm
  };

  double x = 1.4;
  Options options;  // These are common options
  const auto &sum = Optimize(x, loss, options);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(x == Approx(2.0).margin(1e-5));
}

TEST_CASE("tinyopt_simple") { TestSimpleLM(); }

TEST_CASE("explicit template instantiation links across translation units") {
  double parameter = 1.0;
  tinyopt::Options options;
  options.log.enable = false;
  const PrecompiledResidualFunction residuals = &PrecompiledResiduals;

  const auto sum = tinyopt::Optimize(parameter, residuals, options);

  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(parameter == Approx(2.0).margin(1e-3));
}

TEST_CASE("explicit template instantiation of Optimizer class") {
  float parameter = 1.0f;
  tinyopt::Options options;
  options.log.enable = false;
  options.gd.lr = 1.0f;

  auto loss = [&](const auto &x, auto &grad) {
    float res = x - 2.0f;
    // Manually update the H and gradient (J is 1 here)
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad(0) = res;
    }
    return std::abs(res);  // Returns the error norm
  };

  using Optimizer = tinyopt::gd::Optimizer<tinyopt::Vec1f>;
  Optimizer opt(options);
  auto sum = opt(parameter, loss);

  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(parameter == Approx(2.0).margin(1e-3));
}