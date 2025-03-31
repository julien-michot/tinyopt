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

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/norms.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::norms;

void TestScalarNorms() {
  {
    float x = 7;
    REQUIRE(L2(x) == Approx(7).margin(1e-8));
    REQUIRE(L1(x) == Approx(7).margin(1e-8));
    REQUIRE(Linf(x) == Approx(7).margin(1e-8));
    // TODO Check jacobians
  }
}

void TestVecNorms() {
  {
    Vec4 x(1, 2, 3, -4);
    REQUIRE(L2(x) == Approx(std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3])).margin(1e-8));
    REQUIRE(L1(x) == Approx(10).margin(1e-8));
    REQUIRE(Linf(x) == Approx(3).margin(1e-8));
    // TODO Check jacobians
  }
}

TEST_CASE("tinyopt_norms") {
  TestScalarNorms();
  TestVecNorms();
}