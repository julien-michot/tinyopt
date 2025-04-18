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

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

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

  double x = 1;
  REQUIRE(diff::CheckGradient(x, loss));
  nlls::Options options;  // These are common options
  const auto &out = nlls::Optimize(x, loss, options);
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(2.0).margin(1e-5));
}

void TestSimpleGradientDescent() {
  auto loss = [&](const auto &x, auto &grad) {
    double res = x - 2;
    // Manually update the gradient
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad(0) = res;
    }
    return std::abs(res);  // Returns the error norm
  };

  double x = 1;
  REQUIRE(diff::CheckGradient(x, loss));
  gd::Options options;  // These are common options
  const auto &out = gd::Optimize(x, loss, options);
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(2.0).margin(1e-5));
}

TEST_CASE("tinyopt_simple") {
  TestSimpleLM();
  TestSimpleGradientDescent();
}