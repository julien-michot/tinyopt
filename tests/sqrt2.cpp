// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/optimizers/lm.h>
#include <cmath>
#include "tinyopt/diff/gradient_check.h"

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#endif

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::nlls;

inline auto CreateOptions() {
  Options options;
  options.max_iters = 20;
  options.max_consec_failures = 0;
  options.log.enable = true;
  return options;
}

void TestSqrt2(float x0) {
  auto residuals = [&](const auto &x, auto &grad, auto &H) {
    float res = x * x - 2;  // since we want x to be sqrt(2), x*x should be 2
    float J = 2 * x;        // residual's jacobian/derivative w.r.t x
    // Manually update the hessian and gradient
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad(0) = J * res;
      H(0, 0) = J * J;
    }
    // Returns the residual
    return res;
  };

  auto loss = [&](const auto &x, auto &grad, auto &H) {
    auto r = residuals(x, grad, H);
    return r*r;
  };

  float x = x0;
  REQUIRE(diff::CheckResidualsGradient(x, residuals));
  Options options = CreateOptions();
  const auto &out = Optimize(x, loss, options);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(std::abs(x) == Approx(std::sqrt(2.0)).margin(1e-5));
}

void TestSqrt2Jet(double x0) {
  auto loss = [](const auto &x) { return x * x - 2.0; };

  double x = x0;
  Options options = CreateOptions();
  options.cost.use_squared_norm = true;
  options.cost.downscale_by_2 = true;
  const auto &out = Optimize(x, loss, options);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(std::abs(x) == Approx(std::sqrt(2.0)).margin(1e-5));
}

void TestSqrt2Jet2(double x0) {
#if __cplusplus >= 202002L
  auto loss = [&]<typename T>(const T &x) {
#else  // c++17 and below
  auto loss = [&](const auto &x) {
    using T = typename std::decay_t<decltype(x)>;
#endif
    tinyopt::Vector<T, 2> res;
    res[0] = x * x - 2.0;
    res[1] = T(0.1) * (x * x - T(2.0));  // dummy
    return res;
  };

  double x = x0;
  Options options = CreateOptions();
  const auto &out = Optimize(x, loss, options);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(std::abs(x) == Approx(std::sqrt(2.0)).margin(1e-5));
}

void TestSqrt2Jet2GN(double x0) {
  auto loss = [](const auto &x) { return x * x - 2.0; };

  double x = x0;
  Options options = CreateOptions();
  const auto &out = Optimize(x, loss, options);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(std::abs(x) == Approx(std::sqrt(2.0)).margin(1e-5));
}

TEST_CASE("tinyopt_sqrt2") {
  auto x0 = GENERATE(1.0f, -0.3f, 3.2f);
  CAPTURE(x0);
  TestSqrt2(x0);
  TestSqrt2Jet(x0);
  TestSqrt2Jet2(x0);
  if (x0 > 0.0f) TestSqrt2Jet2GN(x0);
}