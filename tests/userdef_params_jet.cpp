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

#include <ostream>

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
template <typename T>  // Template is only needed if you need automatic
                       // differentiation
struct Rectangle {
  using Scalar = T;               // The scalar type
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

  friend std::ostream& operator<<(std::ostream& os, const Rectangle& rect) {
    os << "p1:" << rect.p1.transpose() << ", p2:" << rect.p2.transpose();
    return os;
  }

  Vector<T, Dynamic> p1;  // top left positions (with a dynamic vector to test this)
  Vec2 p2;                              // bottom right positions
};

namespace tinyopt::traits {

// Here we define the parametrization of a Rectangle, this is needed to be able
// to Optimize one.
template <typename T>
struct params_trait<Rectangle<T>> {
  using Scalar = T;               // The scalar type
  static constexpr int Dims = 4;  // Compile-time parameters dimensions

  // Convert a Rectangle to another type 'T2', e.g. T2 = Jet<T>, only used by
  // auto differentiation
  template <typename T2>
  static Rectangle<T2> cast(const Rectangle<T> &rect) {
    return Rectangle<T2>(rect.p1.template cast<T2>(), rect.p2.template cast<T2>());
  }

  // Define update / manifold
  static void pluseq(Rectangle<T> &rect, const Vector<Scalar, Dims> &delta) {
    // Here I'm choosing a non trivial parametrization (delta center x, delta
    // center y, delta width, delta height) just to illustrate one can use any
    // parametrization/manifold
    rect.p1.x() += delta[0];
    rect.p2.x() += delta[0] + delta[2];  // += dx + dw
    rect.p1.y() += delta[1];
    rect.p2.y() += delta[1] + delta[3];  // += dy + dh

    // But you can simply do:
    // rect.p1 += delta.template head<2>();
    // rect.p2 += delta.template tail<2>();
  }
};

}  // namespace tinyopt::traits

using namespace tinyopt;

void TestUserDefinedParameters() {
  using Vec2f = Vector<float, 2>;

  // Let's say I want the rectangle area to be 10*20, the width = 2 * height and
  // the center at (1, 2).
#if __cplusplus >= 202002L
  auto loss = [&]<typename T>(const Rectangle<T> &rect) {
#else // c++17 and below
  auto loss = [&](const auto &rect) {
    using T = typename std::remove_reference_t<decltype(rect)>::Scalar;
#endif
    using std::max;
    using std::sqrt;
    Vector<T, 4> residuals;
    residuals[0] = rect.area() - 10.0f * 20.0f;
    residuals[1] = 100.0f * (rect.width() / max(rect.height(), T(1e-8f)) -
                             2.0f);  // the 1e-8 is to prevent division by 0
    residuals.template tail<2>() = rect.center() - Vector<T, 2>(1, 2);
    return residuals;
  };

  Rectangle<float> rectangle(Vec2f::Zero(), Vec2f::Ones());
  Options options;
  options.solver.damping_init = 1e-1;
  const auto &out = Optimize(rectangle, loss);

  std::cout << "rect:" << "area:" << rectangle.area() << ", c:" << rectangle.center().transpose()
            << ", size:" << rectangle.height() << "x" << rectangle.width()
            << ", loss:" << loss(rectangle).transpose() << "\n";

  REQUIRE(out.Succeeded());
  REQUIRE(rectangle.area() == Approx(10 * 20).margin(1e-5));
  REQUIRE(rectangle.center().x() == Approx(1).margin(1e-5));
  REQUIRE(rectangle.center().y() == Approx(2).margin(1e-5));
  REQUIRE(rectangle.width() == Approx(2 * rectangle.height()).margin(1e-5));
}

TEST_CASE("tinyopt_userdef_params") { TestUserDefinedParameters(); }