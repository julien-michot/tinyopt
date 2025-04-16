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

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::nlls;

using Catch::Approx;

/// Create points on a circle at a regular spacing
Mat2Xf CreateCirle(int n, float r, const Vec2f &center = Vec2f::Zero(), float noise = 0) {
  const float pi = 3.1415926535f;  // not sure why Windows doesn't like M_PI...
  Mat2Xf obs(2, n);
  float angle = 0;
  for (auto o : obs.colwise()) {
    o = center + r * Vec2f(cosf(angle), sinf(angle)) + noise * Vec2f::Random();
    angle += 2 * pi / (n - 1);
  }
  return obs;
}

void TestFitCircle() {
  const float radius = 2;
  const Vec2f center(2, 7);
  const auto obs = CreateCirle(10, radius, center, 1e-5f);

  // loss is the sum of || ||p - center||² - radius² ||
#if __cplusplus >= 202002L
  auto loss = [&obs]<typename Derived>(const MatrixBase<Derived> &x) {
    using T = typename Derived::Scalar;
#else  // c++17 and below
  auto loss = [&](const auto &x) {
    using T = typename std::decay_t<decltype(x)>::Scalar;
#endif
    const auto &center = x.template head<2>();
    const auto radius2 = x.z() * x.z();
    const auto &delta = obs.cast<T>().colwise() - center;
    const auto &residuals = delta.colwise().squaredNorm();
    return (residuals.array() - radius2)
        .matrix()
        .transpose()
        .eval();  // Make sure the returned type is a scalar or Vector
  };

  static_assert(std::is_invocable_v<decltype(loss), const Vec3 &>);

  Vec3 x(0, 0, 1);  // Parametrization: x = {center (x, y), radius}
  Options options;
  options.solver.damping_init = 1e1;  // start closer to a gradient descent
  const auto &out = Optimize(x, loss, options);

  REQUIRE(out.Succeeded());
  REQUIRE(x.x() == Approx(center.x()).margin(1e-5));
  REQUIRE(x.y() == Approx(center.y()).margin(1e-5));
  REQUIRE(x.z() == Approx(radius).margin(1e-5));
}

TEST_CASE("tinyopt_fitcircle") { TestFitCircle(); }