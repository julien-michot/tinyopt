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

#include <type_traits>
#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/solvers/solvers.h>

using Catch::Approx;

using namespace tinyopt;
using namespace tinyopt::solvers;

TEMPLATE_TEST_CASE("tinyopt_solvers2_numdiff", "[solver]", SolverLM<Mat2>, SolverGN<Mat2>,
                   SolverGN<MatX>) {
  TestType solver;
  using Vec = typename TestType::Grad_t;
  SECTION("Resize") { solver.resize(2); }
  SECTION("Solve") {
    Vec x = Vec::Zero(2);
    const Vec2 y = Vec2(4, 5);

    auto loss = [&](const auto &x) { return (x - y).eval(); };

    Vec dx;
    if constexpr (TestType::FirstOrder)
      solver.Solve(x, diff::NumDiff1(x, loss), dx);
    else
      solver.Solve(x, diff::NumDiff2(x, loss), dx);

    REQUIRE(dx[0] == Approx(y[0]).margin(1e-2));
    REQUIRE(dx[1] == Approx(y[1]).margin(1e-2));
  }
}

TEMPLATE_TEST_CASE("tinyopt_solvers1_numdiff", "[solver]", SolverGD<Vec2>) {
  typename TestType::Options options;
  options.lr = 0.1;  // accelerate
  TestType solver(options);
  using Vec = typename TestType::Grad_t;
  SECTION("Resize") { solver.resize(2); }
  SECTION("Solve") {
    Vec x = Vec::Zero(2);
    const Vec2 y = Vec2(4, 5);

    auto loss = [&](const auto &x) { return (x - y).eval(); };

    Vec dx;
    solver.Solve(x, diff::NumDiff1(x, loss), dx);

    REQUIRE(dx[0] == Approx(y[0] * options.lr).margin(1e-2));
    REQUIRE(dx[1] == Approx(y[1] * options.lr).margin(1e-2));
  }
}