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

#define TINYOPT_LOSS_WRAPPER(name, func, expected_code)                                  \
  struct name {                                                                          \
    explicit name(double _th) : th{_th}, th2{_th * _th} {}                               \
    auto operator()(double n, bool export_jac) const { return func(n, th, export_jac); } \
    template <typename T>                                                                \
    auto operator()(const T &x) const {                                                  \
      if constexpr (traits::is_matrix_or_array_v<T>)                                     \
        return func(L2(x), th);                                                          \
      else                                                                               \
        return func(x, th);                                                              \
    }                                                                                    \
    template <typename T>                                                                \
    auto operator()(const T &x, bool export_jac) const {                                 \
      if constexpr (traits::is_matrix_or_array_v<T>)                                     \
        return func(L2(x, export_jac), th);                                              \
      else                                                                               \
        return func(x, th);                                                              \
    }                                                                                    \
    double expected(double n) const { return ((expected_code)); }                        \
    const double th = 0;                                                                 \
    const double th2 = 0;                                                                \
  };

TINYOPT_LOSS_WRAPPER(TruncatedWrapper, Truncated, n > th ? th : n);
TINYOPT_LOSS_WRAPPER(HuberWrapper, Huber, n > th ? sqrt((2.0) * th * n - th * th) : n);
TINYOPT_LOSS_WRAPPER(TukeyWrapper, Tukey,
                     n > th ? (th * sqrt(1.0 - std::pow(1.0 - n * n / th2, 3))) : n);
TINYOPT_LOSS_WRAPPER(ArctanWrapper, Arctan, sqrt(th *std::atan2(n *n, th)));
TINYOPT_LOSS_WRAPPER(CauchyWrapper, Cauchy, th *sqrt(log(1.0 + n * n / th2)));
TINYOPT_LOSS_WRAPPER(GemanMcClureWrapper, GemanMcClure, n / sqrt(n * n + th2));
TINYOPT_LOSS_WRAPPER(BlakeZissermanWrapper, BlakeZisserman,
                     n > th ? (sqrt(-log(exp(-n * n) + exp(-th2)))) : n);

TEMPLATE_TEST_CASE("tinyopt_loss_robust", "[loss]", HuberWrapper) {
  const double th = 1.3;
  TestType loss(th);
  SECTION("Scalar") {
    const double n = 0.5;
    const auto l = loss(n);
    REQUIRE(l == Approx(loss.expected(n)).margin(1e-5));
  }
  SECTION("Scalar Inlier") {
    const double n = 0.3;
    const auto &[l, J] = loss(n, true);
    REQUIRE(l == Approx(loss.expected(n)).margin(1e-5));
    const auto Jad = diff::CalculateJac(n, [&](const auto &n) { return loss(n); });
    TINYOPT_LOG("J:{}, Jad:{}", J, Jad);
    REQUIRE(std::abs(J - Jad[0]) == Approx(0.0).margin(1e-5));
  }
  SECTION("Scalar Outlier") {
    const double n = 2.3;
    const auto &[l, J] = loss(n, true);
    REQUIRE(l == Approx(loss.expected(n)).margin(1e-5));
    const auto Jad = diff::CalculateJac(n, [&](const auto &n) { return loss(n); });
    TINYOPT_LOG("J:{}, Jad:{}", J, Jad);
    REQUIRE(std::abs(J - Jad[0]) == Approx(0.0).margin(1e-5));
  }

  Vec4 x(.1, -0.2, -0.3, 0.4);
  SECTION("Vec Inlier") {
    const double n = x.norm();
    const auto &[l, J] = loss(L2(x, true));
    const auto &[l2, J2] = loss(x, true);
    REQUIRE(l == Approx(loss.expected(n)).margin(1e-5));
    REQUIRE(l2 == Approx(l).margin(1e-5));
    REQUIRE((J - J2).cwiseAbs().maxCoeff() == Approx(0).margin(1e-5));
    const auto Jad = diff::CalculateJac(x, [&](const auto &x) { return loss(x); });
    TINYOPT_LOG("J:{}, Jad:{}", J, Jad);
    REQUIRE((J - Jad).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  SECTION("Vec Outlier") {
    const double th = 0.03;
    TestType loss(th);
    const double n = x.norm();
    const auto &[l, J] = loss(L2(x, true));
    const auto &[l2, J2] = loss(x, true);
    TINYOPT_LOG("l:{} J:{}", l, J);
    REQUIRE(l == Approx(loss.expected(n)).margin(1e-5));
    REQUIRE(l2 == Approx(l).margin(1e-5));
    REQUIRE((J - J2).cwiseAbs().maxCoeff() == Approx(0).margin(1e-5));
    const auto Jad = diff::CalculateJac(x, [&](const auto &x) { return loss(x); });
    TINYOPT_LOG("J:{}, Jad:{}", J, Jad);
    REQUIRE((J - Jad).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}
