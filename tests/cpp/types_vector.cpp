// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

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

void TestVector() {
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

}  // namespace

TEST_CASE("tinyopt_types_vector") { TestVector(); }
