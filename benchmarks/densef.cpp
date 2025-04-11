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
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::nlls::lm;
using namespace tinyopt::losses;

inline auto CreateOptions() {
  Options options;
  options.solver.use_ldlt = true;
  options.max_iters = 5;
  options.log.enable = false;
  options.solver.log.enable = false;
  return options;
}

TEMPLATE_TEST_CASE("Dense", "[benchmark][fixed][dense][float]", Vec3f, Vec6f, VecXf) {
  constexpr Index Dims = TestType::RowsAtCompileTime;
  const Index dims = Dims == Dynamic ? 10 : Dims;
  const TestType y = TestType::Random(dims);
  const TestType stdevs = TestType::Random(dims);  // prior standard deviations
  auto loss = [&](const auto &x) { return MahaWhitened(x - y, stdevs); };
  auto loss2 = [&](const auto &x, auto &grad, auto &H) {
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      const auto &[res, J] = MahaWhitened(x - y, stdevs, true);
      grad = J * res;
      H.diagonal() = stdevs.cwiseInverse().cwiseAbs2();
      return res.norm();               // return √(res.t()*res)
    } else {                           // No gradient
      return MahaNorm(x - y, stdevs);  // return √(res.t()*res)
    }
  };

  const Options options = CreateOptions();

  BENCHMARK("Gaussian Prior [AD]") {
    TestType x = TestType::Random(dims);
    return Optimize(x, loss, options);
  };
  BENCHMARK("Prior") {
    TestType x = TestType::Random(dims);
    return Optimize(x, loss2, options);
  };
}
