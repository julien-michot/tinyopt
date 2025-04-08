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
#include <tinyopt/losses/robust_norms.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::losses;

void TestNorms() {
  SECTION("TruncatedL2") {
    TINYOPT_LOG("** TruncatedL2 Norm");
    SECTION("Scalar Inlier") {
      const float th = 1.2;
      const auto &[s, Js] = TruncatedL2(0.8f, th, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      REQUIRE(s == Approx(0.8f).margin(1e-5));
    }
    SECTION("Scalar Outlier") {
      const float th =  0.2;
      const auto &[s, Js] = TruncatedL2(0.8f, th, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      REQUIRE(s == Approx(th).margin(1e-5));
    }
    SECTION("Vec4 + Jac") {
      const double th = 1.3;
      Vec4 x(1, 2, 3, 4);
      const auto &[s, Js] = TruncatedL2(x, th, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      auto J = diff::CalculateJac(x, [th](const auto x) { return TruncatedL2(x, th); });
      TINYOPT_LOG("Jad:{}", J);
      REQUIRE(s == Approx(th).margin(1e-5));
      REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
    SECTION("Vec4 + Jac") {
      const double th = 10.3;
      Vec4 x(1, 2, 3, 4);
      const auto &[s, Js] = TruncatedL2(x, th, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      auto J = diff::CalculateJac(x, [th](const auto x) { return TruncatedL2(x, th); });
      TINYOPT_LOG("Jad:{}", J);
      REQUIRE(s == Approx(x.norm()).margin(1e-5));
      REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
  }
}

TEST_CASE("tinyopt_robust_norms") { TestNorms(); }
