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

#include <cmath>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "tinyopt/tinyopt.h"

using namespace tinyopt;

using Catch::Approx;

void TestScalars() {
  {
    double x = 1;
    Optimize(x, [](const auto &x) { return x * x - 2.0; });
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
  {
    float x = 1;
    Optimize(x, [](const auto &x) { return x * x - 2.0f; });
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
}

void TestStl() {
  {
    std::array<double, 3> x{{1, 2, 3}};
    Optimize(x, [](const auto &x) { return x[0] + x[1] + x[2] - 10.0; });
    REQUIRE((x[0] + x[1] + x[2]) == Approx(10.0).margin(1e-5));
  }
  {
    std::vector<float> x{{1, 2, 3}};
    Optimize(x, [](const auto &x) { return x[0] + x[1] + x[2] - 10.0f; });
    REQUIRE((x[0] + x[1] + x[2]) == Approx(10.0).margin(1e-5));
  }
}

void TestEigenVector() {
  {
    using Vec = Vec2;
    Vec x = Vec::Ones();
    Optimize(x, [](const auto &x) { return (x - Vec::Constant(2.0)).eval(); });
    REQUIRE((x.array() - 2.0).cwiseAbs().sum() == Approx(0.0).margin(1e-5));
  }
  {
    using Vec = Vec2;
    Vec x = Vec::Ones();
    Optimize(x, [](const auto &x) { return x[0] + x[1] - 10.0; });
    REQUIRE(x[0] + x[1] == Approx(10.0).margin(1e-5));
  }
  {
    using Vec = VecXf;
    Vec x = Vec::Ones(3);
    Optimize(x, [](const auto &x) { return (x.array() - 2.0f).eval(); });
    REQUIRE((x.array() - 2.0f).cwiseAbs().sum() == Approx(0.0).margin(1e-5));
  }
}

void TestEigenMatrix() {
  {
    using Mat = Mat23f;
    Mat x = Mat::Random(), y = Mat::Random() * 10;
    Optimize(x, [&y](const auto &x) {
      using T = typename std::remove_reference_t<decltype(x)>::Scalar;
      return (x - y.template cast<T>()).reshaped().eval(); // Vector
    });
    REQUIRE((x.array() - y.array()).cwiseAbs().sum() == Approx(0.0).margin(1e-5));
  }
  {
    using Mat = Mat32;
    Mat x = Mat::Random(), y = Mat::Random() * 10;
    Optimize(x, [&y](const auto &x) {
      using T = typename std::remove_reference_t<decltype(x)>::Scalar;
      return (x - y.template cast<T>()).eval(); // Matrix
    });
    REQUIRE((x.array() - y.array()).cwiseAbs().sum() == Approx(0.0).margin(1e-5));
  }
  {
    using Mat = Mat3X;
    Mat x = Mat::Random(3, 2), y = Mat::Random(3, 2) * 10;
    const auto &out = Optimize(x, [&y](const auto &x) {
      using T = typename std::remove_reference_t<decltype(x)>::Scalar;
      return (x - y.template cast<T>()).eval(); // Matrix
    });
    REQUIRE((x.array() - y.array()).cwiseAbs().sum() == Approx(0.0).margin(1e-5));
  }
}

TEST_CASE("tinyopt_types") {
  TestScalars();
  TestStl();
  TestEigenVector();
  TestEigenMatrix();
}