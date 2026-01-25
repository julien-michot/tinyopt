// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include "tinyopt/log.h"

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
#include "utils.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::nlls::lm;
using namespace tinyopt::losses;

static const bool enable_log = false;

TEST_CASE("Float", "[benchmark][fixed][scalar]") {
  auto loss = [](const auto &x) { return x * x - 2.0f; };
  Options options = CreateOptions(enable_log);
  options.solver.linear_solver = tinyopt::solvers::Options2::LinearSolver::Inverse;
  options.solver.log.print_failure = true;
  BENCHMARK("√2") {
    float x = Vec1::Random()[0];
    if (enable_log) TINYOPT_LOG("x:{:.12e}", x);
    Optimize(x, loss, options);
  };
}

TEST_CASE("Double", "[benchmark][fixed][scalar]") {
  auto loss = [](const auto &x) { return x * x - 2.0; };
  Options options = CreateOptions(enable_log);
  options.solver.linear_solver = tinyopt::solvers::Options2::LinearSolver::Inverse;
  static StatCounter<double> counter;
  BENCHMARK("√2") {
    double x = Vec1::Random()[0];  // 0.480009157900 fails to converge
    const auto &out = Optimize(x, loss, options);
    counter.AddConv(out.Converged());
    counter.AddFinalIters(out.num_iters);
  };
}

TEMPLATE_TEST_CASE("Dense", "[benchmark][fixed][dense][double]", Vec3, Vec6, Vec12) {
  const TestType y = TestType::Random();
  const TestType stdevs = TestType::Random();  // prior standard deviations
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

  const Options options = CreateOptions(enable_log);
  static StatCounter<TestType> counter;

  BENCHMARK("Prior [AD]") {
    TestType x = TestType::Random();
    Optimize(x, loss, options);
  };
  BENCHMARK("Prior") {
    TestType x = TestType::Random();
    const auto &out = Optimize(x, loss2, options);
    counter.AddConv(out.Converged());
    counter.AddFinalIters(out.num_iters);
  };
}

TEMPLATE_TEST_CASE("Dense", "[benchmark][dyn][dense][double]", VecX) {
  auto dims = GENERATE(3, 6, 12, 33, 50);
  CAPTURE(dims);

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

  const Options options = CreateOptions(enable_log);
  static StatCounter<TestType> counter;

  BENCHMARK("Prior " + std::to_string(dims) + " [AD]") {
    TestType x = TestType::Random(dims);
    Optimize(x, loss, options);
  };
  BENCHMARK("Prior " + std::to_string(dims)) {
    TestType x = TestType::Random(dims);
    const auto &out = Optimize(x, loss2, options);
    counter.AddConv(out.Converged());
    counter.AddFinalIters(out.num_iters);
  };
}
