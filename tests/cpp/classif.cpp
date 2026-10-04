// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <utility>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/auto_diff.h>
#include <tinyopt/diff/num_diff.h>
#include <tinyopt/log.h>
#include <tinyopt/losses/classif.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::losses;

void TestLosses() {
  SECTION("SoftMax scalar") {
    const float x = 0.75f;
    const auto s = Softmax(x);
    TINYOPT_LOG("loss = {}", s);
    REQUIRE(s == Approx(1.0f).margin(1e-6));
  }
  SECTION("SoftMax") {
    Vec4 x = Vec4::Random();
    const auto &[s, Js] = Softmax(x, true);
    TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
    auto J = diff::CalculateJac(x, [](const auto x) { return Softmax(x); });
    TINYOPT_LOG("Jad:{}", J);
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  SECTION("SafeSoftMax scalar") {
    const float x = -2.5f;
    const auto &[s, Js] = SafeSoftmax(x);
    TINYOPT_LOG("loss = [{}, J:{}]", s, Js);
    REQUIRE(s == Approx(1.0f).margin(1e-6));
    REQUIRE(Js == Approx(1.0f).margin(1e-6));
  }
  SECTION("SafeSoftMax") {
    Vec4 x = Vec4::Random();
    const auto &[s, Js] = SafeSoftmax(x, true);
    TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
    auto J = diff::CalculateJac(x, [](const auto x) { return SafeSoftmax(x); });
    TINYOPT_LOG("Jad:{}", J);
    REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}

TEST_CASE("tinyopt_losses_classif") { TestLosses(); }