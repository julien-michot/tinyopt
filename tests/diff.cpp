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
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/auto_diff.h>
#include <tinyopt/diff/num_diff.h>

using Catch::Approx;

using namespace tinyopt;
using namespace tinyopt::diff;

void TestNumDiff1() {
  {
    const Vec3 y_prior = Vec3::Random();
    Vec3 x = Vec3::Zero();
    auto loss = [&](const auto &x) -> Vec3 { return 2 * (x - y_prior); };
    auto loss_nd = NumDiff1(x, loss);

    const Vec3 res = loss(x);
    Vec3 g;
    loss_nd(x, g);
    REQUIRE(g[0] == Approx(2 * res[0]).margin(1e-5));
    REQUIRE(g[1] == Approx(2 * res[1]).margin(1e-5));
    REQUIRE(g[2] == Approx(2 * res[2]).margin(1e-5));
  }
  {
    const Vec3 y_prior = Vec3::Random();
    Vec3 x = Vec3::Zero();
    auto loss = [&](const auto &x) -> Vec2 { return 2 * (x - y_prior).template head<2>(); };
    auto loss_nd = NumDiff1(x, loss);

    const Vec2 res = loss(x);
    Vec3 g;
    loss_nd(x, g);
    REQUIRE(g[0] == Approx(2 * res[0]).margin(1e-5));
    REQUIRE(g[1] == Approx(2 * res[1]).margin(1e-5));
  }
  {
    const float y_prior = 2;
    float x = 0;
    auto loss = [&](const auto &x) { return x - y_prior; };
    auto loss_nd = NumDiff1(x, loss);

    const float res = loss(x);

    Vec1f g;
    loss_nd(x, g);
    REQUIRE(g[0] == Approx(1 * res).margin(1e-3));
  }
}

void TestNumDiff2() {
  {
    const Vec3 y_prior = Vec3::Random();
    Vec3 x = Vec3::Zero();
    auto loss = [&](const auto &x) -> Vec3 { return 2 * (x - y_prior); };
    auto loss_nd = NumDiff2(x, loss);

    const Vec3 res = loss(x);
    Mat3 H;
    Vec3 g;
    loss_nd(x, g, H);
    REQUIRE(g[0] == Approx(2 * res[0]).margin(1e-5));
    REQUIRE(g[1] == Approx(2 * res[1]).margin(1e-5));
    REQUIRE(g[2] == Approx(2 * res[2]).margin(1e-5));
  }
  {
    const float y_prior = 2;
    float x = 0;
    auto loss = [&](const auto &x) { return x - y_prior; };
    auto loss_nd = NumDiff2(x, loss);

    const float res = loss(x);

    Mat1f H;
    Vec1f g;
    loss_nd(x, g, H);
    REQUIRE(g[0] == Approx(1 * res).margin(1e-3));
  }
}

void TestAutoDiff() {
  {
    const float y_prior = 2;
    float x = 0;
    auto loss = [&](const auto &x) { return x - y_prior; };
    auto J = CalculateJac(x, loss);
    REQUIRE(J[0] == Approx(1).margin(1e-3));
  }
  {
    VecX x = VecX::Random(3);
    auto loss = [&](const auto &x) {return 10.0 * x.sum();};
    auto J = CalculateJac(x, loss);
    REQUIRE(J.sum() == Approx(3*10).margin(1e-3));
  }
  {
    const Vec3 y_prior = Vec3::Random();
    Vec3 x = Vec3::Zero();
    auto loss = [&](const auto &x) {
      return (2.0 * (x - y_prior).template head<2>()).eval();
    };
    auto J = CalculateJac(x, loss);
    const Mat23 Je = (Mat23() << 2, 0, 0, 0, 2, 0).finished();
    REQUIRE((Je - J).cwiseAbs().maxCoeff() == Approx(0).margin(1e-3));
  }
}

TEST_CASE("tinyopt_differentiation") {
  TestNumDiff1();
  TestNumDiff2();
  TestAutoDiff();
}