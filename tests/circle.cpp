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

#include "tinyopt/tinyopt.h"

using namespace tinyopt;

using Catch::Approx;
using Vec2f = Eigen::Vector<float, 2>;

/// Create points on a circle at a regular spacing
std::vector<Vec2f> CreateCirle(int n, float r, const Vec2f &center = Vec2f::Zero(), float noise = 0)
{
  std::vector<Vec2f> obs(n);
  float angle = 0;
  for (auto &o : obs) {
    o = center + r * Vec2f(cosf(angle), sinf(angle)) + noise * Vec2f::Random();
    angle += 2 * M_PI / (n - 1);
  }
  return obs;
}

void TestFitCircle() {
  const float radius = 2;
  const Vec2f center(2, 7);
  const auto obs = CreateCirle(10, radius, center, 1e-5);

  // loss is the sum of || ||p - center||² - radius² ||
  auto loss = [&obs]<typename T>(const Eigen::Vector<T, 3> &x) {
    //using T = std::remove_const_t<std::remove_reference_t<decltype(x[0])>>; // recover Jet type
    const auto &center = x.template head<2>();
    const auto radius2 = x.z() * x.z();
    Eigen::Vector<T, Eigen::Dynamic> residuals(obs.size() + 1); // +1 prior
    for (size_t i = 0; i < obs.size(); ++i) {
      residuals(i) = (obs[i].cast<T>() - center).squaredNorm() - radius2;
    }
    // Not really needed but here is how to add a prior on the radius (being 1), with a σ = 1e3
    residuals[obs.size()] = 1e-3 * (x.z() - 1.0);
    return residuals;
  };

  Eigen::Vector<double, 3> x(0, 0, 1); // Parametrization: x = {center (x, y), radius}
  Options options;
  options.damping_init = 1e1; // start closer to a descent gradient
  const auto &out = Optimize(x, loss, options);

  REQUIRE(out.Succeeded());
  REQUIRE(x.x() == Approx(center.x()).epsilon(1e-5));
  REQUIRE(x.y() == Approx(center.y()).epsilon(1e-5));
  REQUIRE(x.z() == Approx(radius).epsilon(1e-5));
}


TEST_CASE("tinyopt_fitcircle") {
  TestFitCircle();
}