// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <ostream>
#include "tinyopt/diff/gradient_check.h"
#include "tinyopt/diff/num_diff.h"
#include "tinyopt/log.h"
#include "tinyopt/math.h"
#include "tinyopt/traits.h"

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

// Example of a rectangle
struct Rectangle {
  using T = double;
  using Vec2 = Vector<T, 2>;  // Just for convenience
  Rectangle() : p1(Vec2::Zero()), p2(Vec2::Zero()) {}
  explicit Rectangle(const Vec2 &_p1, const Vec2 &_p2) : p1(_p1), p2(_p2) {}

  // Returns the area of the rectangle
  T area() const { return (p2 - p1).norm(); }

  // Returns the width of the rectangle
  T width() const { return p2.x() - p1.x(); }
  // Returns the width of the rectangle
  T height() const { return p2.y() - p1.y(); }

  // Returns the center of the rectangle
  Vec2 center() const { return T(0.5) * (p1 + p2); }

  Vec2 p1, p2;  // top left and bottom right positions
};

namespace tinyopt::traits {

// Here we define the parametrization of a Rectangle, this is needed to be able
// to Optimize one.
template <>
struct params_trait<Rectangle> {
  using Scalar = double;
  static constexpr Index Dims = 4;  // Compile-time parameters dimensions
  // Define update / manifold
  static void PlusEq(Rectangle &rect, const Vector<double, Dims> &delta) {
    rect.p1 += delta.template head<2>();
    rect.p2 += delta.template tail<2>();
  }
};

}  // namespace tinyopt::traits

using namespace tinyopt;

void TestUserDefinedParameters() {
  // Let's say I want the rectangle's p1 and p2 to be close to specific points
  auto get_residuals = [&](const Rectangle &rect, auto &grad, auto &H) {
    Vec4 residuals;
    residuals.head<2>() = rect.p1 - Vec2(1, 2);
    residuals.tail<2>() = rect.p2 - Vec2(3, 4);
    // Jacobian (very simple in this case)
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      Mat4 J = Mat4::Identity();
      grad = J.transpose() * residuals;
      if constexpr (!traits::is_nullptr_v<decltype(H)>) H = J.transpose() * J;
    }
    // Returns the norm
    return residuals;
  };

  // Loss is L2² norm of residuals
  auto loss = [&](const Rectangle &rect, auto &grad, auto &H) {
    return get_residuals(rect, grad, H).squaredNorm();
  };

  Rectangle rectangle(Vec2::Zero(), Vec2::Ones());

  REQUIRE(diff::CheckResidualsGradient(rectangle, get_residuals));

  Options options;
  options.lm.damping_init = 1e-1f;
  const auto &out = Optimize(rectangle, loss);

  std::nullptr_t null;

  std::cout << "rect:" << "area:" << rectangle.area() << ", c:" << rectangle.center().transpose()
            << ", size:" << rectangle.height() << "x" << rectangle.width()
            << ", loss:" << loss(rectangle, null, null) << "\n";

  REQUIRE(out.Succeeded());
  REQUIRE(rectangle.p1.x() == Approx(1).margin(1e-5));
  REQUIRE(rectangle.p1.y() == Approx(2).margin(1e-5));
  REQUIRE(rectangle.p2.x() == Approx(3).margin(1e-5));
  REQUIRE(rectangle.p2.y() == Approx(4).margin(1e-5));
}

TEST_CASE("tinyopt_userdef_params") { TestUserDefinedParameters(); }

// Local struct (you can only do that if you don't need Auto. Diff., but ok for Num. Diff.)
template <typename S = double>
struct A {
  using Scalar = S;
  using Vec = Vector<Scalar, 2>;
  static constexpr Index Dims = 2;

  A() : v(Vec::Random() + Vec::Constant(1.0)) {}
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

TEST_CASE("tinyopt_nonlocal_userdef_params") {
  auto residuals = [&](const auto &x, auto &g, auto &H) {
    const Vec2 res = 3.0 * x.v.array() + 2.3;
    if constexpr (!traits::is_nullptr_v<decltype(g)>) {
      const auto Jt = Vec2::Constant(3.0).asDiagonal();
      g = Jt * res;
      H = Vec2::Constant(9).asDiagonal();
    }
    return res;
  };
  A x;
  REQUIRE(diff::CheckResidualsGradient(x, residuals));

  Options options;
  options.lm.damping_init = 1e-5;
  Optimize(x, residuals, options);

  REQUIRE((x.v - Vec2::Constant(-2.3 / 3.0)).norm() == Approx(0.0).margin(1e-5));
}

TEST_CASE("tinyopt_local_userdef_params") {
  // Local struct (you can only do that if you don't need Auto. Diff.)
  struct B {
    using Scalar = double;
    B() : v(Vec2::Random()) {}
    // No Dims (because it's a local struct)
    int dims() const { return 2; }
    // No cast<>() (because it's a local struct and not needed in this example)
    B &operator+=(const Vec2 &delta) {
      v += delta;
      return *this;
    }
    Vec2 v;
  };

  auto residuals = [&](const auto &x, auto &g, auto &H) {
    const Vec2 res = 3.0 * x.v.array() + 2.3;
    if constexpr (!traits::is_nullptr_v<decltype(g)>) {
      const auto Jt = Vec2::Constant(3.0).asDiagonal();
      g = Jt * res;
      H = Vec2::Constant(9).asDiagonal();
    }
    return res;
  };
  B x;
  auto J = diff::EstimateNumJac(x, residuals);
  TINYOPT_LOG("J:{}", J);

  Options options;
  options.lm.damping_init = 1e-5;
  Optimize(x, residuals, options);

  REQUIRE((x.v - Vec2::Constant(-2.3 / 3.0)).norm() == Approx(0.0).margin(1e-5));
}