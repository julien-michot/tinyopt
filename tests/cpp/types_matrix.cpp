// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>

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

void TestMatrix() {
  {
    using Mat = Mat23f;
    Mat x = Mat::Random(), y = Mat::Random() * 10;
    Optimize(x, [&y](const auto &x) {
      using T = typename std::decay_t<decltype(x)>::Scalar;
      return (x - y.template cast<T>()).reshaped().eval();  // Vector
    });
    REQUIRE((x.array() - y.array()).cwiseAbs().sum() == Approx(0.0).margin(1e-5));
  }
  {
    using Mat = Mat32;
    Mat x = Mat::Random(), y = Mat::Random() * 10;
    Optimize(x, [&y](const auto &x) {
      using T = typename std::decay_t<decltype(x)>::Scalar;
      return (x - y.template cast<T>()).eval();  // Matrix
    });
    REQUIRE((x.array() - y.array()).cwiseAbs().sum() == Approx(0.0).margin(1e-5));
  }
  {
    using Mat = Mat3X;
    Mat x = Mat::Random(3, 2), y = Mat::Random(3, 2) * 10;
    const auto &sum = Optimize(x, [&y](const auto &x) {
      using T = typename std::decay_t<decltype(x)>::Scalar;
      return (x - y.template cast<T>()).eval();  // Matrix
    });
    REQUIRE((x.array() - y.array()).cwiseAbs().sum() == Approx(0.0).margin(1e-5));
  }
}

void TestStlMatrix() {
  {
    std::array<Vec2f, 3> x{{Vec2f::Random(), Vec2f::Random(), Vec2f::Random()}};
    Optimize(x, [](const auto &x, auto &grad, auto &H) {
      Vec2f res = x[0] + x[1] + x[2] - Vec2f::Constant(10.0);
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        Matrix<float, 2, 6> J;
        J << 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1;
        H = J.transpose() * J;       // 6x6
        grad = J.transpose() * res;  // 6x1
      }
      return res;
    });
    REQUIRE((x[0] + x[1] + x[2] - Vec2f::Constant(10)).norm() == Approx(0.0).margin(1e-5));
  }
  // NOTE Automatic differentiation not supported on nested types nor on Dynamic sized scalar (.e.g
  // array<VecX, N>)
}

}  // namespace

TEST_CASE("tinyopt_types_matrix") { TestMatrix(); }

TEST_CASE("tinyopt_types_stl_matrix") { TestStlMatrix(); }
