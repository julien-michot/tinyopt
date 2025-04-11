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
#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/auto_diff.h>
#include <tinyopt/log.h>
#include <tinyopt/losses/mahalanobis.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::losses;

template <typename Scalar, int Dims>
auto CreateCov(int dims = Dims, Scalar eps = 1e-2) {
  using Mat = Matrix<Scalar, Dims, Dims>;
  Mat A = Mat::Random(dims, dims);
  Mat C = A * A.transpose() + eps * Mat::Identity();
  return C;
}

void TestSquaredMahalanobis() {
  SECTION("Scalar") {
    float x = 0.8f, var = 2.0;
    const auto &[s, Js] = SquaredMahaNorm(x, var, true);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    REQUIRE(s == Approx(x * x / var).margin(1e-5));
    auto J = diff::CalculateJac(x, [&](const auto x) { return SquaredMahaNorm(x, var); });
    TINYOPT_LOG("Jad:{}", J);
    REQUIRE(std::abs(J[0] - Js) == Approx(0.0).margin(1e-5));
  }
  SECTION("Vec4 + Vars") {
    Vec4 x = Vec4::Random();
    const Vec4 vars = Vec4(1, 2, 4, 16);
    const auto &[s, Js] = SquaredMahaNorm(x, vars, true);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    auto J = diff::CalculateJac(x, [&](const auto x) { return SquaredMahaNorm(x, vars); });
    TINYOPT_LOG("Jad:{}", J);
    const auto expected = x.transpose() * vars.cwiseInverse().asDiagonal() * x;
    REQUIRE(s == Approx(expected).margin(1e-5));
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  SECTION("Vec4 + Cov") {
    Vec4 x = Vec4::Random();
    const auto cov = CreateCov<double, 4>();
    const auto &[s, Js] = SquaredMahaNorm(x, cov, true);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    auto J = diff::CalculateJac(x, [&](const auto x) { return SquaredMahaNorm(x, cov); });
    TINYOPT_LOG("Jad:{}", J);
    const auto expected = x.transpose() * InvCov(cov).value() * x;
    if (std::abs(s - expected) > 1e-5) {
      TINYOPT_LOG("diff too big:{}", std::abs(s - expected));
      TINYOPT_LOG("x:{}", x);
      TINYOPT_LOG("cov:{}", cov);
    }
    REQUIRE(s == Approx(expected).margin(1e-5));
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}

void TestMahalanobis() {
  SECTION("Scalar") {
    float x = 0.8f, var = 2.0;
    const auto &[s, Js] = MahaNorm(x, var, true);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    REQUIRE(s == Approx(std::sqrt(x * x / var)).margin(1e-5));
    auto J = diff::CalculateJac(x, [&](const auto x) { return MahaNorm(x, var); });
    TINYOPT_LOG("Jad:{}", J);
    REQUIRE(std::abs(J[0] - Js) == Approx(0.0).margin(1e-5));
  }
  SECTION("Vec4 + Vars") {
    VecX x = VecX::Random(4);
    const Vec4 vars = Vec4(1, 2, 4, 16);
    const auto &[s, Js] = MahaNorm(x, vars, true);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    auto J = diff::CalculateJac(x, [&](const auto x) { return MahaNorm(x, vars); });
    TINYOPT_LOG("Jad:{}", J);
    const auto expected = std::sqrt(x.transpose() * vars.cwiseInverse().asDiagonal() * x);
    REQUIRE(s == Approx(expected).margin(1e-5));
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  SECTION("Vec4 + Cov") {
    Vec4 x = Vec4::Random();
    const auto cov = CreateCov<double, 4>();
    const auto &[s, Js] = MahaNorm(x, cov, true);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    auto J = diff::CalculateJac(x, [&](const auto x) { return MahaNorm(x, cov); });
    TINYOPT_LOG("Jad:{}", J);
    const auto expected = std::sqrt(x.transpose() * InvCov(cov).value() * x);
    if (std::abs(s - expected) > 1e-5) {
      TINYOPT_LOG("diff too big:{}", std::abs(s - expected));
      TINYOPT_LOG("x:{}", x);
      TINYOPT_LOG("cov:{}", cov);
    }
    REQUIRE(s == Approx(expected).margin(1e-5));
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}

void TestMahaWhitened() {
  SECTION("Vec2 + Vars") {
    VecXf x = VecXf::Random(2);
    const Vec2f stdevs = Vec2f(1, 5);
    const auto &[s, Js] = MahaWhitened(x, stdevs, true);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    auto J = diff::CalculateJac(x, [&](const auto x) { return MahaWhitened(x, stdevs); });
    TINYOPT_LOG("Jad:{}", J);
    const auto expected = (stdevs.cwiseInverse().asDiagonal() * x).eval();
    REQUIRE((s - expected).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));

    const MatXf cov = stdevs.cwiseAbs2().asDiagonal();
    const MatXf U = cov.inverse().llt().matrixU();
    const auto expected_norm = x.transpose() * cov.inverse() * x;
    auto z = MahaWhitenedInfoU(x, U);
    if (std::abs(z.squaredNorm() - expected_norm) > 1e-5) {
      TINYOPT_LOG("diff too big:{}", std::abs(z.squaredNorm() - expected_norm));
      TINYOPT_LOG("x:{}", x);
      TINYOPT_LOG("cov:{}", cov);
    }
    REQUIRE(std::abs(z.squaredNorm() - expected_norm) == Approx(0.0).margin(1e-5));
  }
  SECTION("Vec4 + Cov") {
    Vec4 x = Vec4::Random();
    const auto cov = CreateCov<double, 4>();
    const auto &[y, Js] = MahaWhitened(x, cov, true);
    TINYOPT_LOG("loss = [{}, J:{}]", y, Js);
    auto J = diff::CalculateJac(x, [&](const auto x) { return MahaWhitened(x, cov); });
    TINYOPT_LOG("Jad:{}", J);
    const auto expected_norm = x.transpose() * cov.inverse() * x;
    if (std::abs(y.squaredNorm() - expected_norm) > 1e-5) {
      TINYOPT_LOG("diff too big:{}", std::abs(y.squaredNorm() - expected_norm));
      TINYOPT_LOG("x:{}", x);
      TINYOPT_LOG("cov:{}", cov);
    }
    REQUIRE(std::abs(y.squaredNorm() - expected_norm) == Approx(0.0).margin(1e-5));
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));

    const Mat4 U = cov.inverse().llt().matrixU();
    auto z = MahaWhitenedInfoU(x, U);
    REQUIRE(std::abs(z.squaredNorm() - expected_norm) == Approx(0.0).margin(1e-5));
  }
}

TEST_CASE("tinyopt_loss_mahalanobis") {
  //std::srand(std::time(nullptr));
  TestSquaredMahalanobis();
  TestMahalanobis();
  TestMahaWhitened();
}