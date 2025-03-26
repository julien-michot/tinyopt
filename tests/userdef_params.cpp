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
#include <cstddef>
#include <ostream>

#include <Eigen/Eigen>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "tinyopt/tinyopt.h"

using Catch::Approx;

// Example of a rectangle
struct Rectangle {
  using T = double;
  using Vec2 = Eigen::Vector<T, 2>; // Just for convenience
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

  Vec2 p1, p2; // top left and bottom right positions
};

namespace tinyopt::traits {

// Here we define the parametrization of a Rectangle, this is needed to be able
// to Optimize one.
template <>
struct params_trait<Rectangle> {
  using Scalar = double;
  static constexpr int Dims = 4; // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static constexpr int dims(const Rectangle &) {
    return Dims;
  } // same as Dims
  // Conversion to string
  static std::string toString(const Rectangle &rect) {
    std::stringstream os;
    os << "p1:" << rect.p1.transpose() << ", p2:" << rect.p2.transpose();
    return os.str();
  }

  // Define update / manifold
  static void pluseq(Rectangle &rect,
                     const Eigen::Vector<double, Dims> &delta) {
    rect.p1 += delta.template head<2>();
    rect.p2 += delta.template tail<2>();
  }
};

} // namespace tinyopt::traits

using namespace tinyopt;

void TestUserDefinedParameters() {
  using Vec2 = Eigen::Vector<double, 2>;
  using Vec4 = Eigen::Vector<double, 4>;
  using Mat4 = Eigen::Matrix<double, 4, 4>;

  // Let's say I want the rectangle's p1 and p2 to be close to specific points
  auto loss = [&](const Rectangle &rect, auto &JtJ, auto &Jt_res) {
    Vec4 residuals;
    residuals.head<2>() = rect.p1 - Vec2(1, 2);
    residuals.tail<2>() = rect.p2 - Vec2(3, 4);
    // Jacobian (very simple in this case)
    if constexpr (!traits::is_nullptr_type_v<decltype(JtJ)>) {
      Mat4 J = Mat4::Identity();
      JtJ = J.transpose() * J;
      Jt_res = J.transpose() * residuals;
    }
    // Returns the squared error
    return residuals.squaredNorm();
  };

  Rectangle rectangle(Vec2::Zero(), Vec2::Ones());
  Options options;
  options.damping_init = 1e-1;
  const auto &out = Optimize(rectangle, loss);

  std::nullptr_t null;

  std::cout << "rect:" << "area:" << rectangle.area()
            << ", c:" << rectangle.center().transpose()
            << ", size:" << rectangle.height() << "x" << rectangle.width()
            << ", loss:" << loss(rectangle, null, null) << "\n";

  REQUIRE(out.Succeeded());
  REQUIRE(rectangle.p1.x() == Approx(1).margin(1e-5));
  REQUIRE(rectangle.p1.y() == Approx(2).margin(1e-5));
  REQUIRE(rectangle.p2.x() == Approx(3).margin(1e-5));
  REQUIRE(rectangle.p2.y() == Approx(4).margin(1e-5));
}

TEST_CASE("tinyopt_userdef_params") { TestUserDefinedParameters(); }