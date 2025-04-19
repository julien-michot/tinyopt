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
  gd::Options options;  // These are common options
  options.max_iters = 1000; // let's say it's not the fastest optimizer...
  options.min_error = 0;
  options.min_rerr_dec = 0;
  options.solver.lr = 0.01; // especially with this!
  const auto &out = gd::Optimize(x, loss, options);
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(42.0).margin(1e-5));
}

TEST_CASE("tinyopt_unconstrained") { TestSimpleGradientDescent(); }