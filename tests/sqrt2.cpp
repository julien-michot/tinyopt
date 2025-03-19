// Copyright (C) 2025 Julien Michot. All Rights reserved.

#include <cmath>
#include <utility>

#include "tinyopt/tinyopt.h"

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

using namespace tinyopt::lm;

using Catch::Approx;

void TestSqrt2() {

  using Vec1 = Eigen::Vector<double, 1>;
  Vec1 x(1);

  auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
    float res = x[0]*x[0] - 2; // since we want x to be sqrt(2), x*x should be 2
    float J = 2*x[0]; // residual's jacobian/derivative w.r.t x
    // Manually update the JtJ and Jt*err
    JtJ(0, 0) = J*J;
    Jt_res(0) = J * res;
    // Return both the squared error and the number of residuals (here, we have only one)
    return std::make_pair(res*res, 1);
  };

  const auto &out = LM(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(x[0] == Approx(std::sqrt(2.0)).epsilon(1e-5));
}

TEST_CASE("tinyopt_sqrt2") {
  TestSqrt2();
}