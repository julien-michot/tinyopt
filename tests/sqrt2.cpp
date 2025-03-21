// Copyright (C) 2025 Julien Michot. All Rights reserved.

#include <cmath>
#include <type_traits>
#include <utility>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "tinyopt/tinyopt.h"

using namespace tinyopt::lm;

using Catch::Approx;

void TestSqrt2() {

  auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
    double res = x * x - 2; // since we want x to be sqrt(2), x*x should be 2
    double J = 2 * x; // residual's jacobian/derivative w.r.t x
    // Manually update the JtJ and Jt*err
    JtJ(0, 0) = J * J;
    Jt_res(0) = J * res;
    // Return both the squared error and the number of residuals (here, we have only one)
    return std::make_pair(res*res, 1);
  };

  float x = 1;
  const auto &out = LM(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(x == Approx(std::sqrt(2.0)).epsilon(1e-5));
}

void TestSqrt2Jet() {

  auto loss = [&](const auto &x) {
    return 1.0 * x * x - 2.0;
  };

  double x = 1;
  const auto &out = AutoLM(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(x == Approx(std::sqrt(2.0)).epsilon(1e-5));
}

void TestSqrt2Jet2() {

  auto loss = [&](const auto &x) {
    // needed by Ceres's Jet (alternative is to use a templated lambda)
    using T = std::remove_const_t<std::remove_reference_t<decltype(x)>>;
    tinyopt::Vector<T, 2> res;
    res[0] = x * x - 2.0;
    res[1] = T(0.1) * (x * x - T(2.0)); // dummy
    return res;
  };

  double x = 1;
  const auto &out = AutoLM(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(x == Approx(std::sqrt(2.0)).epsilon(1e-5));
}

TEST_CASE("tinyopt_sqrt2") {
  TestSqrt2();
  TestSqrt2Jet();
  TestSqrt2Jet2();
}