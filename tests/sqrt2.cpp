// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <utility>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "tinyopt/tinyopt.h"

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

void TestSqrt2() {
  auto loss = [&](const auto &x, auto &grad, auto &H) {
    double res = x * x - 2;  // since we want x to be sqrt(2), x*x should be 2
    double J = 2 * x;        // residual's jacobian/derivative w.r.t x
    // Manually update the H and Jt*err
    H(0, 0) = J * J;
    grad(0) = J * res;
    // Returns the error
    return std::sqrt(res * res);
    // You can also return the error (scaled or not) as well as the number of residuals
    // return std:.make_pair(res*res, 1);
  };

  float x = 1;
  const auto &out = lm::Optimize(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
}

void TestSqrt2Jet() {
  auto loss = [](const auto &x) { return x * x - 2.0; };

  double x = 1;
  const auto &out = lm::Optimize(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
}

void TestSqrt2Jet2() {
#if __cplusplus >= 202002L
  auto loss = [&]<typename T>(const T &x) {
#else  // c++17 and below
  auto loss = [&](const auto &x) {
    using T = typename std::remove_const_t<std::remove_reference_t<decltype(x)> >;
#endif
    tinyopt::Vector<T, 2> res;
    res[0] = x * x - 2.0;
    res[1] = T(0.1) * (x * x - T(2.0));  // dummy
    return res;
  };

  double x = 1;
  const auto &out = lm::Optimize(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
}

void TestSqrt2Jet2GN() {
  auto loss = [](const auto &x) { return x * x - 2.0; };

  double x = 1;
  const auto &out = gn::Optimize(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
}

TEST_CASE("tinyopt_sqrt2") {
  TestSqrt2();
  TestSqrt2Jet();
  TestSqrt2Jet2();
  TestSqrt2Jet2GN();
}