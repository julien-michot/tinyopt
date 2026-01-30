// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

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
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::lm;
using namespace tinyopt::losses;

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
      return res.squaredNorm();               // return √(res.t()*res)
    } else {                                  // No gradient
      return MahaSquaredNorm(x - y, stdevs);  // return √(res.t()*res)
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
