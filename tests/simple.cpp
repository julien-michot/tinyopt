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

#include "tinyopt/tinyopt.h"

using namespace tinyopt;

using Catch::Approx;

void TestSimple() {
  auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
    double res = x - 2;
    // Manually update the JtJ and Jt*err (J is 1 here)
    JtJ(0, 0) = 1;
    Jt_res(0) = res;
    // Returns the squared error
    return res*res;
  };

  double x = 1;
  const auto &out = Optimize(x, loss);
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(2.0).margin(1e-5));
  std::cout << "Stop reason: " << out.StopReasonDescription() << "\n";
}

TEST_CASE("tinyopt_simple") {
  TestSimple();
}