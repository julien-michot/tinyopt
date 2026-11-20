// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tinyopt/tinyopt.h>

#include "bundle_adjustment.h"
#include "bundle_adjustment_tinyopt.h"
#include "mt_benchmark.h"
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::benchmark::bundle_adjustment;

TEST_CASE("BA block-sparse", "[benchmark][bundle-adjustment][sparse]") {
  const auto dimensions =
      GENERATE(std::pair{2, 200}, std::pair{4, 600}, std::pair{10, 3000}, std::pair{100, 50000});
  const Problem initial_problem = MakeProblem(dimensions.first, dimensions.second);
  Problem verification_problem = initial_problem;
  const BlockSparseLoss loss;
  Options options = CreateBundleAdjustmentOptions();
  options.stop.max_iters = 100;
  const double reference_initial_cost = ReferenceReprojectionCost(initial_problem);

  using BlockHessian = BlockSparseLoss::Hessian;
  lm::Optimizer<BlockHessian> verification_optimizer(options);
  const auto& verification = verification_optimizer(verification_problem, loss);
  std::nullptr_t null_gradient{};
  BlockHessian unused_hessian;
  const double initial_cost = loss(initial_problem, null_gradient, unused_hessian).cost;
  REQUIRE(verification.Succeeded());
  REQUIRE(verification.Converged());
  REQUIRE(initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost.cost < initial_cost * 0.1);
  REQUIRE(verification.final_cost.cost < MaxFinalCost(initial_problem));
  REQUIRE(verification_problem.initial_cameras[0] == initial_problem.initial_cameras[0]);
  REQUIRE(verification_problem.initial_points[0] == initial_problem.initial_points[0]);
  const std::string label = ProblemLabel(initial_problem);
  tinyopt::benchmark::PrintIterations("Bundle adjustment", label, "tinyopt", verification.num_iters,
                                      verification.Converged());

  BENCHMARK(std::string(label)) {
    Problem problem = initial_problem;
    lm::Optimizer<BlockHessian> optimizer(options);
    const auto& result = optimizer(problem, loss);
    return result.final_cost.cost;
  };
}

#if defined(TINYOPT_USE_OPENMP)
// This benchmark runs the complete LM solve: OpenMP parallelizes the CPU block-sparse
// accumulation, while the block-sparse linear solver and parameter updates remain on the host.
TEST_CASE("BA block-sparse multithread", "[benchmark][mt]") {
  const Problem problem = MakeProblem(100, 50000);
  const BlockSparseLoss loss;
  struct Result {
    double cost;
    int iterations;
  };
  auto optimize = [&](std::uint16_t num_threads) {
    Problem state = problem;
    Options options = CreateBundleAdjustmentOptions();
    options.num_threads = num_threads;
    options.stop.max_consec_failures = 3;
    loss.set_num_threads(num_threads);
    lm::Optimizer<BlockSparseLoss::Hessian> optimizer(options);
    const auto& result = optimizer(state, loss);
    REQUIRE(result.Succeeded());
    REQUIRE(result.Converged());
    return Result{result.final_cost.cost, result.num_iters};
  };

  const auto [serial, serial_ms] =
      tinyopt::benchmark::MeasureMilliseconds([&] { return optimize(1); });
  const auto [parallel, parallel_ms] =
      tinyopt::benchmark::MeasureMilliseconds([&] { return optimize(8); });
  REQUIRE(parallel.cost == Catch::Approx(serial.cost).margin(1e-12));
  tinyopt::benchmark::PrintThreadComparison("tinyopt", serial_ms, parallel_ms, serial.cost,
                                            parallel.cost);
  std::printf("MT_ITERATIONS backend=tinyopt one_thread=%d eight_threads=%d\n", serial.iterations,
              parallel.iterations);

  BENCHMARK("1 thread") { return optimize(1).cost; };
  BENCHMARK("8 threads") { return optimize(8).cost; };
}

TEST_CASE("Indexed factor accumulation multithread", "[mt]") {
  constexpr Index kFactorCount = 16384;
  const Vector<double, 3> target(0.2, -0.1, 0.3);
  auto residual_and_jacobian = [&](Index index, const Vector<double, 3>& value) {
    const double a = 0.1 + 0.1 * static_cast<double>(index % 19);
    const double b = 0.2 + 0.1 * static_cast<double>(index % 23);
    const double phase_x = value[0] + a;
    const double phase_y = value[1] + b;
    const double residual = std::sin(phase_x) + std::cos(phase_y) + value[2] -
                            std::sin(target[0] + a) - std::cos(target[1] + b) - target[2];
    const Vector<double, 3> jacobian(std::cos(phase_x), -std::sin(phase_y), 1.0);
    return std::pair{residual, jacobian};
  };
  auto factor = [&](Index index, const Vector<double, 3>& value, Vector<double, 3>& gradient,
                    Matrix<double, 3, 3>& hessian) {
    const auto [residual, jacobian] = residual_and_jacobian(index, value);
    gradient.noalias() += jacobian * residual;
    hessian.noalias() += jacobian * jacobian.transpose();
    return Cost(0.5 * residual * residual, 1);
  };
  auto cost = [&](const Vector<double, 3>& value, auto& gradient) {
    double total = 0.0;
    for (Index index = 0; index < kFactorCount; ++index) {
      const auto [residual, jacobian] = residual_and_jacobian(index, value);
      if constexpr (!traits::is_nullptr_v<decltype(gradient)>)
        gradient.noalias() += jacobian * residual;
      total += 0.5 * residual * residual;
    }
    return total;
  };
  const Vector<double, 3> initial_value = Vector<double, 3>::Zero();
  REQUIRE(diff::CheckGradient(initial_value, cost, 1e-4, diff::Method::kCentral, false));
  auto optimize = [&](std::uint16_t num_threads) {
    Vector<double, 3> value = Vector<double, 3>::Zero();
    Options options;
    options.num_threads = num_threads;
    options.exp.measure_time = false;
    options.exp.save_history = false;
    options.log.enable = false;
    const Summary result = OptimizeFactors(value, kFactorCount, factor, options);
    REQUIRE(result.Succeeded());
    REQUIRE(result.Converged());
    REQUIRE((value - target).norm() < 1e-6);
    return result.final_cost.cost;
  };

  const auto [serial_cost, serial_ms] =
      tinyopt::benchmark::MeasureMilliseconds([&] { return optimize(1); });
  const auto [parallel_cost, parallel_ms] =
      tinyopt::benchmark::MeasureMilliseconds([&] { return optimize(8); });
  REQUIRE(parallel_cost == Catch::Approx(serial_cost).margin(1e-12));
  tinyopt::benchmark::PrintThreadComparison("tinyopt-factors", serial_ms, parallel_ms, serial_cost,
                                            parallel_cost);

  BENCHMARK("1 thread") { return optimize(1); };
  BENCHMARK("8 threads") { return optimize(8); };
}
#endif

TEST_CASE("BA sparse", "[benchmark][bundle-adjustment][sparse]") {
  const auto dimensions = GENERATE(std::pair{2, 200}, std::pair{4, 600}, std::pair{10, 3000});
  const Problem initial_problem = MakeProblem(dimensions.first, dimensions.second);
  Problem verification_problem = initial_problem;
  const SparseLoss loss;
  Options options = CreateBundleAdjustmentOptions();
  options.stop.max_iters = 100;
  const double reference_initial_cost = ReferenceReprojectionCost(initial_problem);

  using Hessian = SparseLoss::Hessian;
  lm::Optimizer<Hessian> verification_optimizer(options);
  const auto& verification = verification_optimizer(verification_problem, loss);
  std::nullptr_t null_gradient{};
  Hessian unused_hessian;
  const double initial_cost = loss(initial_problem, null_gradient, unused_hessian).cost;
  REQUIRE(verification.Succeeded());
  REQUIRE(verification.Converged());
  REQUIRE(initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost.cost < initial_cost * 0.1);
  REQUIRE(verification.final_cost.cost < MaxFinalCost(initial_problem));
  REQUIRE(verification_problem.initial_cameras[0] == initial_problem.initial_cameras[0]);
  REQUIRE(verification_problem.initial_points[0] == initial_problem.initial_points[0]);
  tinyopt::benchmark::PrintIterations("Bundle adjustment",
                                      ProblemLabel(initial_problem) + " sparse", "tinyopt",
                                      verification.num_iters, verification.Converged());

  const std::string label = ProblemLabel(initial_problem) + " sparse";
  BENCHMARK(std::string(label)) {
    Problem problem = initial_problem;
    lm::Optimizer<Hessian> optimizer(options);
    const auto& result = optimizer(problem, loss);
    return result.final_cost.cost;
  };
}