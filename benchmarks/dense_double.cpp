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
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::lm;

TEMPLATE_TEST_CASE("tinyopt_bench_scalar", "[benchmark][fixed][scalar]", double) {
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

TEMPLATE_TEST_CASE("tinyopt_bench_dense_fixed", "[benchmark][fixed][dense][double]", Vec2, Vec4,
                   Vec6) {
  constexpr int Dims = TestType::RowsAtCompileTime;
  const TestType y = TestType::Random();
  const TestType stdevs = TestType::Random();  // prior standard deviations
  auto loss = [&](const auto &x) { return loss::MahDiag(x - y, stdevs); };
  auto loss2 = [&](const auto &x, auto &grad, auto &H) {
    Matrix<double, Dims, Dims> J = Matrix<double, Dims, Dims>::Identity();
    const TestType res = loss::MahDiag(x - y, stdevs, &J);
    grad = J * res;
    H.diagonal() = stdevs.cwiseInverse().cwiseAbs2();  // or Jt*J
    return std::sqrt(res.dot(res));                    // return √(res.t()*res)
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

TEMPLATE_TEST_CASE("tinyopt_bench_dense_dyn", "[benchmark][dyn][dense]", VecX) {
  constexpr int N = 10;
  const TestType y = TestType::Random(N);
  const TestType stdevs = TestType::Random(N);  // prior standard deviations
  auto loss = [&](const auto &x) { return loss::MahDiag(x - y, stdevs); };
  auto loss2 = [&](const auto &x, auto &grad, auto &H) {
    MatX J = MatX::Identity(N, N);
    const VecX res = loss::MahDiag(x - y, stdevs, &J);
    grad = J * res;
    H.diagonal() = stdevs.cwiseInverse().cwiseAbs2();  // or Jt*J
    return std::sqrt(res.dot(res));                    // return √(res.t()*res)
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