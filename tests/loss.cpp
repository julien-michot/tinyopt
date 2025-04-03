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

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/loss.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::loss;

/*
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
  SECTION("Mahalanobis [Vec3]") {
    const Mat3 C = Vec3(3, 2, 3).asDiagonal();
    const Vec3 Ja_num = EstimateJac(a, [&b, &C](const auto &a) {
      return std::sqrt((a - b).transpose() * C.inverse() * (a - b));
    });
    const Vec3 Jb_num = EstimateJac(b, [&a, &C](const auto &b) {
      return std::sqrt((a - b).transpose() * C.inverse() * (a - b));
    });
    Vec3 Ja, Jb;  // Jacobians
    Mahalanobis(a, b, C, &Ja, &Jb);
    REQUIRE((Ja - Ja_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    REQUIRE((Jb - Jb_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}*/

void TestActivationsScalar() {
  SECTION("Sigmoid on scalar") {
    const float x = 2;
    const float a = Sigmoid(x);
    std::cout << "a:" << a << "\n";
    REQUIRE(a == Approx(0.880797).margin(1e-8));
  }
  SECTION("Sigmoid on scalar + jac") {
    const float x = 2;
    Vec3 J = Vec3::Ones();
    const auto &[a, Ja] = Sigmoid(x, J);
    std::cout << "a:" << a << "\n";
    std::cout << "Ja:" << Ja.transpose() << "\n";
    REQUIRE(a == Approx(0.880797).margin(1e-8));
    // Check Jacobian
    const auto Ja_num = tinyopt::diff::EstimateJac(a, [](auto &x) { return Sigmoid(x); });
    Vec3 Ja_num_x_J = J * Ja_num[0];
    std::cout << "Ja_num_x_J:" << Ja_num_x_J.transpose() << "\n";
    REQUIRE((Ja_num_x_J - Ja).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }
}

void TestActivationsVector() {
  SECTION("Sigmoid on a vector") {
    const Vec2 x(0.1, 0.2);
    Mat23 J = Mat23::Identity();
    const auto &[a, Ja] = Sigmoid(x, J);
    std::cout << "a:" << a << "\n";
    std::cout << "Ja:" << Ja << "\n";
    REQUIRE(a[0] == Approx(0).margin(1e-8));
    // Check Jacobian
    const auto Ja_num = tinyopt::diff::EstimateJac(a, [](auto &x) { return Sigmoid(x); });
    Mat23 Ja_num_x_J = Ja_num * J;
    std::cout << "Ja_num_x_J:" << Ja_num_x_J << "\n";
    REQUIRE((Ja_num_x_J - Ja).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }
}

void TestMahalanobis() {
  SECTION("Mahalanobis norm with standard deviations") {
    const Vec2 x(0.1, 0.2);
    const Vec2 stdevs(1, 2);
    Mat2 J = Mat2::Identity();
    const double expected_norm = x.transpose() * stdevs.cwiseAbs2().cwiseInverse().asDiagonal() * x;
    const Vec2 xs = MahaDiag(x, stdevs, &J);  // scaled x
    REQUIRE(xs.squaredNorm() == Approx(expected_norm).margin(1e-8));
    REQUIRE((J.diagonal() - stdevs.cwiseInverse()).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }

  SECTION("Mahalanobis norm with a full covariance matrix") {
    const Vec2 x(0.1, 0.2);
    Mat2 C;  // prior covariance
    C << 10, 2, 2, 4;
    Mat2 J = Mat2::Identity();
    const double expected_norm = x.transpose() * C.inverse() * x;
    const Vec2 xs = Maha(x, C, &J);  // scaled x
    REQUIRE(xs.squaredNorm() == Approx(expected_norm).margin(1e-8));

    Mat2 JtJ = J.transpose() * J;
    REQUIRE((JtJ - C.inverse()).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }

  // Mahalanobis norm with an Information Matrix (sqrt upper)
  SECTION("Mahalanobis norm with an Information Matrix (sqrt upper)") {
    const Vec2 x(0.1, 0.2);
    Mat2 C;  // prior covariance
    C << 10, 2, 2, 4;
    Mat2 J = Mat2::Identity();
    const double expected_norm = x.transpose() * C.inverse() * x;
    const Mat2 U = C.inverse().llt().matrixU();
    const Vec2 xs = MahaInfoU(x, U, &J);  // scaled x
    REQUIRE(xs.squaredNorm() == Approx(expected_norm).margin(1e-8));

    Mat2 JtJ = J.transpose() * J;
    REQUIRE((JtJ - C.inverse()).cwiseAbs().maxCoeff() == Approx(0).margin(1e-8));
  }
}

TEST_CASE("tinyopt_norms") {
  // TestScalarNorms();
  // TestVecNorms();
  TestActivationsScalar();
  TestActivationsVector();
  TestMahalanobis();
}