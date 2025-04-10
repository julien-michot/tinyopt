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
#endif

#include <tinyopt/diff/auto_diff.h>
#include <tinyopt/log.h>
#include <tinyopt/losses/norms.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::losses;

#define LOSS_WRAPPER(name, func, expected_code)          \
  struct name {                                          \
    name() {}                                            \
    template <typename T>                                \
    auto operator()(const T &x) const {                  \
      return func(x);                                    \
    }                                                    \
    template <typename T>                                \
    auto operator()(const T &x, bool export_jac) const { \
      return func(x, export_jac);                        \
    }                                                    \
    template <typename T>                                \
    auto expected(const T &x) const {                    \
      if constexpr (traits::is_matrix_or_array_v<T>)     \
        return ((expected_code));                        \
      else                                               \
        return expected(Vector<T, 1>(x));                \
    }                                                    \
  };

LOSS_WRAPPER(L1Wrapper, L1, x.template lpNorm<1>());
LOSS_WRAPPER(L2Wrapper, L2, x.norm());
LOSS_WRAPPER(SquaredL2Wrapper, SquaredL2, x.squaredNorm());
LOSS_WRAPPER(LinfWrapper, Linf, x.template lpNorm<Infinity>());

TEMPLATE_TEST_CASE("tinyopt_loss_norms", "[loss]", L1Wrapper, L2Wrapper, SquaredL2Wrapper,
                   LinfWrapper) {
  TestType loss;
  SECTION("Scalar") {
    float x = 0.8f;
    const auto &[s, Js] = loss(x, true);
    TINYOPT_LOG("loss = [{}, J:{}] exp:{}", s, Js, loss.expected(x));
    REQUIRE(s == Approx(loss.expected(x)).margin(1e-5));
    auto J = diff::CalculateJac(x, [&](const auto x) { return loss(x); });
    TINYOPT_LOG("Jad:{}", J);
    REQUIRE(std::abs(J[0] - Js) == Approx(0.0).margin(1e-5));
  }
  SECTION("Vec4 + Jac") {
    Vec4 x = Vec4::Random();
    const auto &[s, Js] = loss(x, true);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    auto J = diff::CalculateJac(x, [&](const auto x) { return loss(x); });
    TINYOPT_LOG("Jad:{}", J);
    REQUIRE(s == Approx(loss.expected(x)).margin(1e-5));
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}

template <typename Scalar, int Dims>
auto CreateCov(int dims = Dims, Scalar eps = 1e-6) {
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
    Vec4 x = Vec4::Random();
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
    REQUIRE(s == Approx(expected).margin(1e-5));
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}

TEST_CASE("tinyopt_loss_maha_norm") {
  TestSquaredMahalanobis();
  TestMahalanobis();
}