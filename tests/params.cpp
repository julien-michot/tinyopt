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

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#endif

#include <tinyopt/math.h>
#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::nlls;

/// Create points on a circle at a regular spacing
inline Mat2X CreateCirle(int n, float r, const VecX &center = Vec2::Zero(), float noise = 0) {
  const float pi = 3.1415926535f;  // not sure why Windows doesn't like M_PI...
  Mat2X obs(2, n);
  float angle = 0;
  for (auto o : obs.colwise()) {
    o = center + r * Vec2(cosf(angle), sinf(angle)) + noise * Vec2::Random();
    angle += 2 * pi / (n - 1);
  }
  return obs;
}

TEST_CASE("tinyopt_params_pack") {
  float radius = 1.0;
  VecX center = Vec2(0, 0);
  const auto obs = CreateCirle(10, radius, center, 1e-5f);

  ParamsPack2<VecX&, float&> ps(center, radius); // TODO AVOID COPY

  auto loss = [&](const auto &ps) {
    const auto &center = ps.p0();
    const auto radius2 = ps.p1() * ps.p1();
    using T = decltype(ps.p0()[0]);
    const auto &delta = obs.template cast<T>().colwise() - center;
    const auto &residuals = delta.colwise().squaredNorm();
    return (residuals.array() - radius2)
        .matrix()
        .transpose()
        .eval();  // Make sure the returned type is a scalar or Vector
  };

  Options options;
  const auto &out = Optimize(ps, loss, options);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
}