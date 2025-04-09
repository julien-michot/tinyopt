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
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/losses/losses.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::loss;

void TestLossOnScalars() {
  {
    float x = 7;
    REQUIRE(L2(x) == Approx(x).margin(1e-8));
    // TODO Check jacobians
  }
}

void TestLossOnVectors() {
  {
    Vec4 x(1, 2, 3, -4);
    REQUIRE((L2(x) - x).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }
  // Mahalanobis norm with standard deviations
  {
    const Vec2 x(0.1, 0.2);
    const Vec2 stdevs(1, 2);
    Mat2 J = Mat2::Identity();
    const double expected_norm = x.transpose() * stdevs.cwiseAbs2().cwiseInverse().asDiagonal() * x;
    const Vec2 xs = MahDiag(x, stdevs, &J);  // scaled x
    REQUIRE(xs.squaredNorm() == Approx(expected_norm).margin(1e-8));
    REQUIRE((J.diagonal() - stdevs.cwiseInverse()).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }
  // Mahalanobis norm with a full covariance matrix
  {
    const Vec2 x(0.1, 0.2);
    Mat2 C;  // prior covariance
    C << 10, 2, 2, 4;
    Mat2 J = Mat2::Identity();
    const double expected_norm = x.transpose() * C.inverse() * x;
    const Vec2 xs = Mah(x, C, &J);  // scaled x
    REQUIRE(xs.squaredNorm() == Approx(expected_norm).margin(1e-8));

    Mat2 JtJ = J.transpose() * J;
    REQUIRE((JtJ - C.inverse()).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }
  // Mahalanobis norm with an Information Matrix (sqrt upper)
  {
    const Vec2 x(0.1, 0.2);
    Mat2 C;  // prior covariance
    C << 10, 2, 2, 4;
    Mat2 J = Mat2::Identity();
    const double expected_norm = x.transpose() * C.inverse() * x;
    const Mat2 U = C.inverse().llt().matrixU();
    const Vec2 xs = MahInfoU(x, U, &J);  // scaled x
    REQUIRE(xs.squaredNorm() == Approx(expected_norm).margin(1e-8));

    Mat2 JtJ = J.transpose() * J;
    REQUIRE((JtJ - C.inverse()).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }
}

TEST_CASE("tinyopt_loss") {
  TestLossOnScalars();
  TestLossOnVectors();
}