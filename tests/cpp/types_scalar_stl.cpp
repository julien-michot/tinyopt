// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <vector>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::nlls;

namespace {

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

}  // namespace

TEST_CASE("tinyopt_types_scalars") { TestScalars(); }

TEST_CASE("tinyopt_types_stl") { TestStl(); }
