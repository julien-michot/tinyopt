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

#include <cmath>
#include <ostream>

#include <Eigen/Eigen>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "tinyopt/tinyopt.h"

using namespace tinyopt;

using Catch::Approx;


// Example of a rectangle
template <typename T> struct Rectangle {
  using Vec2 = Eigen::Vector<T, 2>; // Just for convenience
  explicit Rectangle() : p1(Vec2::Zero()), p2(Vec2::Zero()) {}

  // Returns the area of the rectangle
  T area() const { return (p2 - p1).norm(); }

  // Returns the width of the rectangle
  T width() const { return p2.x() - p1.x(); }
  // Returns the width of the rectangle
  T height() const { return p2.y() - p1.y(); }

  // Returns the center of the rectangle
  Vec2 center() const { return T(0.5) * (p1 + p2); }

  // Define how to print the class [NEEDED]
  friend std::ostream& operator<<(std::ostream& os, const Rectangle& rect) {
    os << "p1:" << rect.p1.transpose() << ", p2:" << rect.p2.transpose() << std::endl;
    return os;
  }
  Vec2 p1, p2; // top left and bottom right positions
};


// Example of a rectangle with parametrization
template <typename T> struct RectangleParams : Rectangle<T> {
  using Scalar = T; // Scalar (will be replaced by a Jet if automatic differentiation is used)
  static constexpr int Dims = 4; // Number of dimensions (compile time), here 4

  explicit RectangleParams() : Rectangle<T>() {}

  // Convert a Rectangle to another type 'T2', e.g. T2 = Jet<T>, used by auto differentiation
  template <typename T2>
  RectangleParams<T2> cast() const {
    RectangleParams<T2> rect;
    rect.p1 = this->p1.template cast<T2>();
    rect.p2 = this->p2.template cast<T2>();
    return rect;
  }

  // Define update / manifold
  RectangleParams& operator+=(const Eigen::Vector<T, 4>& delta) {
    // Here I'm choosing a non trivial parametrization (delta center x, delta center y, delta width, delta height)
    // It would have been simpler to do p1 += delta.head<2>(), p2 += delta.tail<2>() but I want
    // to illustrate a different parametrization.
    this->p1.x() += delta[0];
    this->p2.x() += delta[0] + delta[2]; // += dx + dw
    this->p1.y() += delta[1];
    this->p2.y() += delta[1] + delta[3]; // += dy + dh
    return *this;
  }

  // Returns the rectangle dimensions at execution time (here same as at compile time) [NEEDED]
  int dims() const { return Dims; }
};


void TestUserDefinedParameters1() {

  // Let's say I want the rectangle area to be 10*20, the width = 2 * height and
  // the center at (1, 2).
  auto loss = [&]<typename T>(const Rectangle<T> &rect) {
    Eigen::Vector<T, 3> residuals;
    residuals[0] = rect.area() - 10 * 20;
    residuals[1] = rect.width() / (rect.height() + 1e-8) - 2; // the 1e-8 is to prevent division by 0
    residuals[2] = (rect.center() - Eigen::Vector<T, 2>(1, 2)).squaredNorm();
    return residuals;
  };

  RectangleParams<float> rectangle;
  const auto &out = Optimize(rectangle, loss);
  REQUIRE(out.Succeeded());
  REQUIRE(rectangle.area() == Approx(10 * 20).epsilon(1e-5));
  REQUIRE(rectangle.center().x() == Approx(1).epsilon(1e-5));
  REQUIRE(rectangle.center().y() == Approx(2).epsilon(1e-5));
  REQUIRE(rectangle.width() == Approx(2 * rectangle.height()).epsilon(1e-5));
}

// TODO test with Rectangle + only traits/specializations
// ideally I just define cast + operator+=(const Eigen::Vector<T, 4>& delta) are recover the Dims!

TEST_CASE("tinyopt_userdef_params") {
  TestUserDefinedParameters1();
  // TODO TestUserDefinedParameters2
}