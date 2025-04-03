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

#include <cstddef>
#include "tinyopt/diff/num_diff.h"
#include "tinyopt/math.h"
#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/optimizers/nlls.h>
#include <tinyopt/params.h>

using namespace tinyopt;
using namespace tinyopt::nlls;

template <typename T>
inline auto SimpleExp(const T &v) {
  using Scalar = typename T::Scalar;
  Eigen::Matrix<Scalar, 3, 3> h;
  h << Scalar(1), -v(2), v(1), v(2), Scalar(1), -v(0), -v(1), v(0), Scalar(1);
  return h;
}

void TestParams() {
  SECTION("1 param w manifold") {
    Mat3 R = Mat3::Identity();
    auto manifold = [](auto &R, const Vec3 &w) { R *= SimpleExp(w); };
    auto x = CreateParams(R, 3, manifold);
    REQUIRE(x.x == R);
  }
}

void TestNumDiffParams() {
  SECTION("Numerical Diff Fixed Dims") {
    Mat3 R = Mat3::Identity();
    auto manifold = [](auto &R, const auto &w) { R *= SimpleExp(w); };
    auto x = CreateParams<3>(R, manifold);
    REQUIRE(x.x == R);

    const Mat3 prior_inv = SimpleExp(Vec3(7, 8, 9));
    auto loss = [&](auto &R) { return (R * prior_inv).norm(); };
    {
      auto loss_nd = diff::NumDiff1(x, loss);
      Vec3 g;
      double e = loss_nd(x, g);
      REQUIRE(e == Catch::Approx((Mat3::Identity() * prior_inv).norm()).margin(1e-3));
    }
    {
      auto loss_nd = diff::NumDiff2(x, loss);
      Vec3 g;
      Mat3 H;
      double e = loss_nd(x, g, H);
      REQUIRE(e == Catch::Approx((Mat3::Identity() * prior_inv).norm()).margin(1e-3));
    }
  }

  SECTION("Numerical Diff Dynamic Dims") {
    MatXf R = Mat3f::Identity();
    auto manifold = [](auto &R, const auto &w) { R *= SimpleExp(w); };
    auto x = CreateParams(R, 3, manifold);
    REQUIRE(x.x == R);

    const Mat3f prior_inv = SimpleExp(Vec3f(7, 8, 9));
    auto loss = [&](auto &R) { return (R * prior_inv).norm(); };
    {
      auto loss_nd = diff::NumDiff1(x, loss);
      VecXf g;
      double e = loss_nd(x, g);
      REQUIRE(e == Catch::Approx((Mat3f::Identity() * prior_inv).norm()).margin(1e-3));
    }
    {
      auto loss_nd = diff::NumDiff2(x, loss);
      VecXf g;
      MatXf H;
      double e = loss_nd(x, g, H);
      REQUIRE(e == Catch::Approx((Mat3f::Identity() * prior_inv).norm()).margin(1e-3));
    }
  }
}

void TestAutoDiffParams() {
  SECTION("Auto Diff Fixed Dims") {
    Mat3 R = Mat3::Identity();
    auto manifold = [](auto &R, const auto &w) { R *= SimpleExp(w); };
    auto x = CreateParams<3>(R, manifold);
    REQUIRE(x.x == R);

    const Mat3 prior_inv = SimpleExp(Vec3(7, 8, 9));
    auto loss = [&](auto &R) { return (R * prior_inv).norm(); };

    const auto optimize = [&](auto &x, const auto &func, const auto &) {
      Vec3 g;
      Mat3 H;
      return func(x, g, H);
    };

    double e = OptimizeJet(x, loss, optimize, std::nullptr_t{});
    REQUIRE(e == Catch::Approx((Mat3::Identity() * prior_inv).norm()).margin(1e-3));
  }
}

void TestOptimizeParams() {
  SECTION("Optimize R*hat(x)") {
    Mat3 R = Mat3::Identity();
    auto manifold = [](auto &R, const auto &w) { R *= SimpleExp(w); };
    auto x = CreateParams<3>(R, manifold);
    REQUIRE(x.x == R);

    const Mat3 prior_inv = SimpleExp(Vec3(7, 8, 9));
    auto loss = [&](auto &R) {
      using T = typename std::remove_const_t<std::remove_reference_t<decltype(x)> >::Scalar;
      return (R * prior_inv.template cast<T>()).eval();
    };

    Options options;  // These are common options
    options.log.print_mean_x = true;
    options.log.print_J_jet = true;
    const auto &out = Optimize(x, loss, options);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE((R * prior_inv).cwiseAbs().maxCoeff() == Catch::Approx(0.0).margin(1e-5));
    std::cout << "Stop reason: " << out.StopReasonDescription() << "\n";
  }
}

TEST_CASE("tinyopt_params") {
  TestParams();
  TestNumDiffParams();
  TestAutoDiffParams();
  TestOptimizeParams();
}
