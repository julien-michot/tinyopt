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

#include <iostream>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#endif

#include <tinyopt/diff/auto_diff.h>
#include <tinyopt/log.h>
#include <tinyopt/losses/norms.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::losses;

// TODO do same as in robust_norms.cpp, use a struct and TEMPLATE_TEST_CASE

void TestNorms() {
  SECTION("L1") {
    TINYOPT_LOG("** L1 Norm");
    SECTION("Scalar") {
      const auto &[s, Js] = L1(0.8f, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
    }
    SECTION("Vec4 + Jac") {
      Vec4 x = Vec4::Random();
      const auto &[s, Js] = L1(x, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      auto J = diff::CalculateJac(x, [](const auto x) { return L1(x); });
      TINYOPT_LOG("Jad:{}", J);
      REQUIRE(s == Approx(x.lpNorm<1>()).margin(1e-5));
      REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
  }

  SECTION("L2") {
    TINYOPT_LOG("** L2 Norm");
    SECTION("Scalar") {
      const auto &[s, Js] = L2(0.8f, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
    }
    SECTION("Vec4 + Jac") {
      Vec4 x = Vec4::Random();
      const auto &[s, Js] = L2(x, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      auto J = diff::CalculateJac(x, [](const auto x) { return L2(x); });
      TINYOPT_LOG("Jad:{}", J);
      REQUIRE(s == Approx(x.norm()).margin(1e-5));
      REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
  }

  SECTION("L∞") {
    TINYOPT_LOG("** L∞ Norm");
    SECTION("Scalar") {
      const auto &[s, Js] = Linf(0.8f, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
    }
    SECTION("Vec4 + Jac") {
      Vec4 x = Vec4::Random();
      const auto &[s, Js] = Linf(x, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      auto J = diff::CalculateJac(x, [](const auto x) { return Linf(x); });
      TINYOPT_LOG("Jad:{}", J);
      REQUIRE(s == Approx(x.lpNorm<Infinity>()).margin(1e-5));
      REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
  }
}

TEST_CASE("tinyopt_norms") { TestNorms(); }
