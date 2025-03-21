// Copyright (C) 2025 Julien Michot. All Rights reserved.

#include <ceres/jet.h>
#include <cmath>
#include <type_traits>
#include <utility>

#include "ceres/jet.h"

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "tinyopt/tinyopt.h"

using namespace tinyopt::lm;

using Catch::Approx;

void TestSqrt2() {

  using Vec1 = Eigen::Vector<double, 1>;
  Vec1 x(1);

  auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
    double res = x[0]*x[0] - 2; // since we want x to be sqrt(2), x*x should be 2
    double J = 2*x[0]; // residual's jacobian/derivative w.r.t x
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

void TestSqrt2Jet() {
  using Vec1 = Eigen::Vector<double, 1>;

  // Simpler interface
  auto loss = [&](const auto &x) {
    // needed by Ceres's Jet (alternative is to use a templated lambda)
    //using T = std::remove_const_t<std::remove_reference_t<decltype(x[0])>>;
    using T = std::remove_reference_t<decltype(x)>::Scalar;
    //return Eigen::Vector<T, 1>(x * x - Eigen::Vector<T, 1>(2)); // so verbose...
    return Eigen::Vector<T, 1>(x[0] * x[0] - T(2)); // so verbose...
    //return x * x - T(2); // so verbose...
  };

  Vec1 x(1);
  const auto &out = AutoLM(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(x[0] == Approx(std::sqrt(2.0)).epsilon(1e-5));
}

TEST_CASE("tinyopt_sqrt2") {
  TestSqrt2();
  TestSqrt2Jet();
}