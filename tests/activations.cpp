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

#include <tinyopt/diff/auto_diff.h>
#include <tinyopt/diff/num_diff.h>
#include <tinyopt/log.h>
#include <tinyopt/losses/activations.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::loss::activations;

void TestLosses() {
  SECTION("Sigmoid") {
    TINYOPT_LOG("** Sigmoid")
    TINYOPT_LOG("loss = {}", Sigmoid(0.3f));
    TINYOPT_LOG("loss = {}", Sigmoid(Vec2f::Random()));
    SECTION("Sigmoid + Jac") {
      {
        const auto &[s, Js] = Sigmoid(std::make_pair(0.5f, true));
        TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      }
      {
        const auto &[s, Js] = Sigmoid(Vec2f::Random(), true);
        TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      }
      {
        const auto &[s, Js] = Sigmoid(0.2f, true);
        TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      }
      {
        Vec4 x = Vec4::Random();
        const auto &[s, Js] = Sigmoid(x, true);
        TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
        auto J = diff::CalculateJac(x, [](const auto x) { return Sigmoid(x); });
        TINYOPT_LOG("Jad:{}", J);
        REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
      }
    }
  }

  SECTION("Tanh") {
    TINYOPT_LOG("** Tanh")
    TINYOPT_LOG("loss = {}", Tanh(0.3));
    TINYOPT_LOG("loss = {}", Tanh(Vec2f::Random()));

    SECTION("Tanh + Jac") {
      const auto &[s, Js] = Tanh(Vec2f::Random(), true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      {
        Vec4 x = Vec4::Random();
        const auto &[s, Js] = Tanh(x, true);
        TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
        auto J = diff::CalculateJac(x, [](const auto x) { return Tanh(x); });
        TINYOPT_LOG("J:{}", J);
        REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
      }
    }
  }

  SECTION("ReLU") {
    TINYOPT_LOG("** ReLU")
    TINYOPT_LOG("loss = {}", ReLU(0.3));
    TINYOPT_LOG("loss = {}", ReLU(Vec2f::Random()));

    SECTION("ReLU + Jac") {
      const auto &[s, Js] = ReLU(Vec2f::Random(), true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      {
        Vec4 x = Vec4::Random();
        const auto &[s, Js] = ReLU(x, true);
        TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
        auto J = diff::CalculateJac(x, [](const auto x) { return ReLU(x); });
        TINYOPT_LOG("J:{}", J);
        REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
      }
    }
  }

  SECTION("LeakyReLU") {
    TINYOPT_LOG("** LeakyReLU")
    const float a = 0.6;
    TINYOPT_LOG("loss = {}", LeakyReLU(0.3, a));
    TINYOPT_LOG("loss = {}", LeakyReLU(Vec2f::Random(), a));

    SECTION("LeakyReLU + Jac") {
      Vec4f x = Vec4f::Random();
      const auto &[s, Js] = LeakyReLU(x, a, true);
      TINYOPT_LOG("loss = [{}, \nJ:{}]", s, Js);
      auto J = diff::CalculateJac(x, [a](const auto x) { return LeakyReLU(x, a); });
      TINYOPT_LOG("J:{}", J);
      REQUIRE((J - Js).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
  }
}

TEST_CASE("tinyopt_losses_activations") { TestLosses(); }