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
#include "tinyopt/math.h"

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/distances.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::distances;
using namespace tinyopt::diff;

void TestDistances() {
  {
    float a = 7, b = 9;
    SECTION("Euclidean [Scalar]") { REQUIRE(Euclidean(a, b) == Approx(2).margin(1e-8)); }
    SECTION("Manhattan [Scalar]") { REQUIRE(Manhattan(a, b) == Approx(2).margin(1e-8)); }
    float C = 9;
    const double exp_mah = std::sqrt((a - b) * (a - b) / C);
    SECTION("Mahalanobis [Vec3]") { REQUIRE(Mahalanobis(a, b, C) == Approx(exp_mah).margin(1e-8)); }
  }
  {
    const Vec3 a(1, 2, 3), b(1, -4, 4);
    SECTION("Euclidean [Vec3]") { REQUIRE(Euclidean(a, b) == Approx((a - b).norm()).margin(1e-8)); }
    SECTION("Manhattan [Vec3]") {
      REQUIRE(Manhattan(a, b) == Approx((a - b).cwiseAbs().sum()).margin(1e-8));
    }
    SECTION("Cosine [Vec3]") {
      const double exp_cosine = a.normalized().dot(b.normalized());
      REQUIRE(Cosine(a, b) == Approx(exp_cosine).margin(1e-8));
    }
    SECTION("Mahalanobis [Vec3]") {
      const Mat3 C = Vec3(3, 2, 3).asDiagonal();
      const double exp_mah = std::sqrt((a - b).transpose() * C.inverse() * (a - b));
      REQUIRE(Mahalanobis(a, b, C) == Approx(exp_mah).margin(1e-8));
    }
    SECTION("Linf [Vec3]") {
      const double exp_mah = (a - b).lpNorm<Infinity>();
      REQUIRE(Linf(a, b) == Approx(exp_mah).margin(1e-8));
    }
  }
}

void TestDistancesJac() {
  {
    const Vec3 a(1, 2, 3), b(1, -4, 4);
    SECTION("Euclidean [Vec3]") {
      const Vec3 Ja_num = EstimateJac(a, [&b](const auto &a) { return (a - b).norm(); });
      const Vec3 Jb_num = EstimateJac(b, [&a](const auto &b) { return (a - b).norm(); });
      Vec3 Ja, Jb;  // Jacobians
      Euclidean(a, b, &Ja, &Jb);
      REQUIRE((Ja - Ja_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
      REQUIRE((Jb - Jb_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
    SECTION("Manhattan [Vec3]") {
      const Vec3 Ja_num = EstimateJac(
          a, [&b](const auto &a) { return (a - b).cwiseAbs().sum(); }, Method::kCentral);
      const Vec3 Jb_num = EstimateJac(
          b, [&a](const auto &b) { return (a - b).cwiseAbs().sum(); }, Method::kCentral);
      Vec3 Ja, Jb;  // Jacobians
      REQUIRE((Ja - Ja_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
      REQUIRE((Jb - Jb_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
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
    SECTION("Linf [Vec3]") {
      const Vec3 Ja_num =
          EstimateJac(a, [&b](const auto &a) { return (a - b).template lpNorm<Infinity>(); });
      const Vec3 Jb_num =
          EstimateJac(b, [&a](const auto &b) { return (a - b).template lpNorm<Infinity>(); });
      Vec3 Ja, Jb;  // Jacobians
      Linf(a, b, &Ja, &Jb);
      REQUIRE((Ja - Ja_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
      REQUIRE((Jb - Jb_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
  }
}

TEST_CASE("tinyopt_loss") {
  TestDistances();
  TestDistancesJac();
}