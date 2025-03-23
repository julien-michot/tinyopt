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
    // Returns the squared error
    return res*res;
    // You can also return the error (scaled or not) as well as the number of residuals
    // return std:.make_pair(res*res, 1);
  };

  float x = 1;
  const auto &out = LM(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(x == Approx(std::sqrt(2.0)).epsilon(1e-5));
}

void TestSqrt2Jet() {

  auto loss = [](const auto &x) {
    return x * x - 2.0;
  };

  double x = 1;
  const auto &out = AutoLM(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(x == Approx(std::sqrt(2.0)).epsilon(1e-5));
}

void TestSqrt2Jet2() {

  auto loss = [&]<typename T>(const T &x) {
    // Alternative for c++17 and below:
    // using T = std::remove_const_t<std::remove_reference_t<decltype(x)>>; for c++17 and below
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