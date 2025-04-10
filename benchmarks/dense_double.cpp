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
#include "tinyopt/losses/mahalanobis.h"

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::nlls::lm;
using namespace tinyopt::losses;

TEMPLATE_TEST_CASE("Scalar", "[benchmark][fixed][scalar]", double) {
  auto loss = [](const auto &x) { return x * x - TestType(2.0); };
  Options options;
  options.solver.use_ldlt = false;
  options.log.enable = false;
  options.solver.log.enable = false;
  BENCHMARK("√2") {
    TestType x = Vec1f::Random()[0];
    return Optimize(x, loss, options);
  };
}

TEMPLATE_TEST_CASE("Dense", "[benchmark][fixed][dense][double]", Vec2, Vec4,
                   Vec6) {
  const TestType y = TestType::Random();
  const TestType stdevs = TestType::Random();  // prior standard deviations
  auto loss = [&](const auto &x) { return MahaWhitened(x - y, stdevs); };
  auto loss2 = [&](const auto &x, auto &grad, auto &H) {
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      const auto &[res, J] = MahaWhitened(x - y, stdevs, true);
      grad = J * res;
      H.diagonal() = stdevs.cwiseInverse().cwiseAbs2();
      return res.norm();                          // return √(res.t()*res)
    } else {                                      // No gradient
      return MahaNorm(x - y, stdevs);  // return √(res.t()*res)
    }
  };

  Options options;
  options.log.enable = false;
  options.solver.log.enable = false;
  BENCHMARK("Gaussian Prior [AD]") {
    TestType x = TestType::Random();
    return Optimize(x, loss, options);
  };
  BENCHMARK("Gaussian Prior") {
    TestType x = TestType::Random();
    return Optimize(x, loss2, options);
  };
}

TEMPLATE_TEST_CASE("Dense", "[benchmark][dyn][dense]", VecX) {
  constexpr int N = 10;
  const TestType y = TestType::Random(N);
  const TestType stdevs = TestType::Random(N);  // prior standard deviations
  auto loss = [&](const auto &x) { return MahaWhitened(x - y, stdevs); };
  auto loss2 = [&](const auto &x, auto &grad, auto &H) {
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      const auto &[res, J] = MahaWhitened(x - y, stdevs, true);
      grad = J * res;
      H.diagonal() = stdevs.cwiseInverse().cwiseAbs2();  // or Jt*J
      return res.norm();                                 // return √(res.t()*res)
    } else {
      return MahaNorm(x - y, stdevs);  // return √(res.t()*res)
    }
  };

  Options options;
  options.log.enable = false;
  options.solver.log.enable = false;
  BENCHMARK("Gaussian Prior [AD]") {
    TestType x = TestType::Random(N);
    return Optimize(x, loss, options);
  };
  BENCHMARK("Gaussian Prior") {
    TestType x = TestType::Random(N);
    return Optimize(x, loss2, options);
  };
}