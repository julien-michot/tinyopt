// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::nlls;

using Catch::Approx;

void TestSimpleLM() {
  auto loss = [&](const auto &x, auto &grad, auto &H) {
    double res = x - 2;
    // Manually update the H and gradient (J is 1 here)
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad(0) = res;
      H(0, 0) = 1;
    }
    return std::abs(res);  // Returns the error norm
  };

  double x = 1.4;
  nlls::Options options;  // These are common options
  const auto &out = nlls::Optimize(x, loss, options);
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x == Approx(2.0).margin(1e-5));
}

TEST_CASE("tinyopt_simple") {
  TestSimpleLM();
}