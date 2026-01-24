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

#include <tinyopt/3rdparty/traits/lieplusplus.h>

using namespace tinyopt;
using namespace tinyopt::nlls;

using Catch::Approx;

void TestPosePriorJet() {
  using Pose = lieplusplus::group::SEn3<double, 1>;

  const Pose prior_inv = Pose::exp(Vec6::Random());

  Pose pose = Pose::exp(Vec6::Random());
  Options options;
  options.log.print_J_jet = false;
  const auto &out = Optimize(
      pose,
      [&](const auto &x) {
        using T = typename std::decay_t<decltype(x)>::Scalar;
        return (traits::params_trait<Pose>::cast<T>(prior_inv) * x).log();
      },
      options);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE((prior_inv * pose).log().norm() == Approx(0.0).margin(1e-5));
}

void TestPosePrior() {
  using Pose = lieplusplus::group::SEn3<double, 1>;

  const Pose prior_inv = Pose::exp(Vec6::Random());

  Pose pose = Pose::exp(Vec6::Random());
  const auto &out = Optimize(pose, [&](const auto &x, auto &grad, auto &H) {
    const auto &res = (prior_inv * x).log();
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      const auto &J = Pose::rightJacobian(res);
      H = J.transpose() * J;
      grad = J.transpose() * res;
    }
    return res.norm();
  });

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE((prior_inv * pose).log().norm() == Approx(0.0).margin(1e-5));
}

TEST_CASE("tinyopt_sophus") {
  TestPosePriorJet();
  TestPosePrior();
}