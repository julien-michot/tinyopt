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
    const auto &out = Optimize(x, [&y](const auto &x) {
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

TEST_CASE("tinyopt_types_scalars") { TestScalars(); }
TEST_CASE("tinyopt_types_stl") { TestStl(); }
TEST_CASE("tinyopt_types_matrix") {
  TestVector();
  TestMatrix();
}
TEST_CASE("tinyopt_types_stl_matrix") { TestStlMatrix(); }
