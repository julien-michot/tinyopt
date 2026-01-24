// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

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

#define LOSS_WRAPPER(name, func, expected_code)                                             \
  struct name {                                                                             \
    explicit name(double _th) : th2{_th * _th} {}                                           \
    auto operator()(double n2, bool export_jac) const { return func(n2, th2, export_jac); } \
    template <typename T>                                                                   \
    auto operator()(const T &x) const {                                                     \
      if constexpr (traits::is_matrix_or_array_v<T>)                                        \
        return func(SquaredL2(x), th2);                                                     \
      else                                                                                  \
        return func(x, th2);                                                                \
    }                                                                                       \
    template <typename T>                                                                   \
    auto operator()(const T &x, bool export_jac) const {                                    \
      if constexpr (traits::is_matrix_or_array_v<T>)                                        \
        return func(SquaredL2(x, export_jac), th2);                                         \
      else                                                                                  \
        return func(x, th2);                                                                \
    }                                                                                       \
    double expected(double n2) const {                                                      \
      const double th = std::sqrt(th2);                                                     \
      const double n = std::sqrt(n2);                                                       \
      (void)n;                                                                              \
      (void)th;                                                                             \
      return ((expected_code));                                                             \
    }                                                                                       \
    const double th2 = 0;                                                                   \
  };

// Ok, let's define all the robust norms

LOSS_WRAPPER(TruncatedWrapper, Truncated, n > th ? th2 : n2);
LOSS_WRAPPER(HuberWrapper, Huber, n > th ? (2.0 * th * n - th2) : n2);
LOSS_WRAPPER(TukeyWrapper, Tukey, n > th ? th2 : (th2 * (1.0 - std::pow(1.0 - n2 / th2, 3.0))));
LOSS_WRAPPER(ArctanWrapper, Arctan, th *std::atan2(n2, th));
LOSS_WRAPPER(CauchyWrapper, Cauchy, th2 *std::log(1.0 + n2 / th2));
LOSS_WRAPPER(GemanMcClureWrapper, GemanMcClure, n2 / (n2 + th2));
LOSS_WRAPPER(BlakeZissermanWrapper, BlakeZisserman, -log(exp(-n2) + exp(-th2)));

// Oh, is that all? then let's test them

TEMPLATE_TEST_CASE("tinyopt_loss_robust", "[loss]", TruncatedWrapper, HuberWrapper, TukeyWrapper,
                   ArctanWrapper, CauchyWrapper, GemanMcClureWrapper, BlakeZissermanWrapper) {
  const double th = 1.3;
  TestType loss(th);
  SECTION("Scalar") {
    const double n2 = 0.5;
    const auto l = loss(n2);
    REQUIRE(l == Approx(loss.expected(n2)).margin(1e-5));
  }
  SECTION("Scalar Inlier") {
    const double n2 = 0.3;
    const auto &[l, J] = loss(n2, true);
    REQUIRE(l == Approx(loss.expected(n2)).margin(1e-5));
    const auto Jad = diff::CalculateJac(n2, [&](const auto &n2) { return loss(n2); });
    TINYOPT_LOG("J:{}, Jad:{}", J, Jad);
    REQUIRE(std::abs(J - Jad[0]) == Approx(0.0).margin(1e-5));
  }
  SECTION("Scalar Outlier") {
    const double n2 = 2.3 * 2.3;
    const auto &[l, J] = loss(n2, true);
    REQUIRE(l == Approx(loss.expected(n2)).margin(1e-5));
    const auto Jad = diff::CalculateJac(n2, [&](const auto &n) { return loss(n); });
    TINYOPT_LOG("J:{}, Jad:{}", J, Jad);
    REQUIRE(std::abs(J - Jad[0]) == Approx(0.0).margin(1e-5));
  }

  Vec4 x(.1, -0.2, -0.3, 0.4);
  SECTION("Vec Inlier") {
    const double n2 = x.squaredNorm();
    const auto &[l, J] = loss(SquaredL2(x, true));
    const auto &[l2, J2] = loss(x, true);
    REQUIRE(l == Approx(loss.expected(n2)).margin(1e-5));
    REQUIRE(l2 == Approx(l).margin(1e-5));
    REQUIRE((J - J2).cwiseAbs().maxCoeff() == Approx(0).margin(1e-5));
    const auto Jad = diff::CalculateJac(x, [&](const auto &x) { return loss(x); });
    TINYOPT_LOG("J:{}, Jad:{}", J, Jad);
    REQUIRE((J - Jad).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  SECTION("Vec Outlier") {
    const double th = 0.03;
    TestType loss(th);
    const double n2 = x.squaredNorm();
    const auto &[l, J] = loss(SquaredL2(x, true));
    const auto &[l2, J2] = loss(x, true);
    TINYOPT_LOG("l:{} J:{}", l, J);
    REQUIRE(l == Approx(loss.expected(n2)).margin(1e-5));
    REQUIRE(l2 == Approx(l).margin(1e-5));
    REQUIRE((J - J2).cwiseAbs().maxCoeff() == Approx(0).margin(1e-5));
    const auto Jad = diff::CalculateJac(x, [&](const auto &x) { return loss(x); });
    TINYOPT_LOG("J:{}, Jad:{}", J, Jad);
    REQUIRE((J - Jad).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}
