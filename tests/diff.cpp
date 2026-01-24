// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/auto_diff.h>
#include <tinyopt/diff/num_diff.h>

using Catch::Approx;

using namespace tinyopt;
using namespace tinyopt::diff;

void TestCreateNumDiffFunc1() {
  {
    const Vec3 y_prior = Vec3::Random();
    Vec3 x = Vec3::Zero();
    auto loss = [&](const auto &x) -> Vec3 { return 2 * (x - y_prior); };
    auto loss_nd = CreateNumDiffFunc1(x, loss);

    const Vec3 res = loss(x);
    Vec3 g;
    loss_nd(x, g);
    REQUIRE(g[0] == Approx(2 * res[0]).margin(1e-5));
    REQUIRE(g[1] == Approx(2 * res[1]).margin(1e-5));
    REQUIRE(g[2] == Approx(2 * res[2]).margin(1e-5));
  }
  {
    const Vec3 y_prior = Vec3::Random();
    Vec3 x = Vec3::Zero();
    auto loss = [&](const auto &x) -> Vec2 { return 2 * (x - y_prior).template head<2>(); };
    auto loss_nd = CreateNumDiffFunc1(x, loss);

    const Vec2 res = loss(x);
    Vec3 g;
    loss_nd(x, g);
    REQUIRE(g[0] == Approx(2 * res[0]).margin(1e-5));
    REQUIRE(g[1] == Approx(2 * res[1]).margin(1e-5));
  }
  {
    const float y_prior = 2;
    float x = 0;
    auto loss = [&](const auto &x) { return x - y_prior; };
    auto loss_nd = CreateNumDiffFunc1(x, loss);

    const float res = loss(x);

    Vec1f g;
    loss_nd(x, g);
    REQUIRE(g[0] == Approx(1 * res).margin(1e-3));
  }
}

void TestCreateNumDiffFunc2() {
  {
    const Vec3 y_prior = Vec3::Random();
    Vec3 x = Vec3::Zero();
    auto loss = [&](const auto &x) -> Vec3 { return 2 * (x - y_prior); };
    auto loss_nd = CreateNumDiffFunc2(x, loss);

    const Vec3 res = loss(x);
    Mat3 H;
    Vec3 g;
    loss_nd(x, g, H);
    REQUIRE(g[0] == Approx(2 * res[0]).margin(1e-5));
    REQUIRE(g[1] == Approx(2 * res[1]).margin(1e-5));
    REQUIRE(g[2] == Approx(2 * res[2]).margin(1e-5));
  }
  {
    const float y_prior = 2;
    float x = 0;
    auto loss = [&](const auto &x) { return x - y_prior; };
    auto loss_nd = CreateNumDiffFunc2(x, loss);

    const float res = loss(x);

    Mat1f H;
    Vec1f g;
    loss_nd(x, g, H);
    REQUIRE(g[0] == Approx(1 * res).margin(1e-3));
  }
}

void TestAutoDiff() {
  {
    const float y_prior = 2;
    float x = 0;
    auto loss = [&](const auto &x) { return x - y_prior; };
    auto J = CalculateJac(x, loss);
    REQUIRE(J[0] == Approx(1).margin(1e-3));
  }
  {
    VecX x = VecX::Random(3);
    auto loss = [&](const auto &x) { return 10.0 * x.sum(); };
    auto J = CalculateJac(x, loss);
    REQUIRE(J.sum() == Approx(3 * 10).margin(1e-3));
  }
  {
    const Vec3 y_prior = Vec3::Random();
    Vec3 x = Vec3::Zero();
    auto loss = [&](const auto &x) { return (2.0 * (x - y_prior).template head<2>()).eval(); };
    auto J = CalculateJac(x, loss);
    const Mat23 Je = (Mat23() << 2, 0, 0, 0, 2, 0).finished();
    REQUIRE((Je - J).cwiseAbs().maxCoeff() == Approx(0).margin(1e-3));
  }
}

void TestNumDiffUserStruct() {
  // Local struct (you can only do that if you don't need Auto. Diff., but ok for Num. Diff.)
  struct A {
    using Scalar = double;
    int dims() const { return 2; }
    A() : v(Vec2::Random() + Vec2::Constant(2.0)) {}

    A &operator+=(const Vec2 &delta) {
      v += delta;
      return *this;
    }
    Vec2 v;
  };
  A a;

  auto residuals = [&](const auto &a) { return (3.0 * a.v).eval(); };
  const auto &[res, J] = NumEval(a, residuals);
  const Mat2 J2 = Vec2(3, 3).asDiagonal();
  REQUIRE((J - J2).cwiseAbs().maxCoeff() == Approx(0).margin(1e-3));
}

// Global struct
template <typename S = double>
struct A {
  using Scalar = S;
  using Vec = Vector<Scalar, 2>;
  static constexpr Index Dims = 2;

  A() : v(Vec::Random()) {}
  A(const Vec &vv) : v(vv) {}

  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const A &a) {
    return A<T2>(a.v.template cast<T2>());
  }

  A &operator+=(const Vec &delta) {
    v += delta;
    return *this;
  }
  Vec v;
};


void TestAutoDiffUserStruct() {
  auto residuals = [&](const auto &a) { return (3.0 * a.v).eval(); };
  A a;
  const auto &[res, J] = Eval(a, residuals);
  Mat2 J2 = Vec2(3, 3).asDiagonal();
  REQUIRE((J - J2).cwiseAbs().maxCoeff() == Approx(0).margin(1e-3));
}

TEST_CASE("tinyopt_ num_diff") {
  TestCreateNumDiffFunc1();
  TestCreateNumDiffFunc2();
}

TEST_CASE("tinyopt_num_diff") { TestNumDiffUserStruct(); }
TEST_CASE("tinyopt_auto_diff") { TestAutoDiffUserStruct(); }
