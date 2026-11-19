// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <utility>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimize_factors.h>
#include <tinyopt/optimize_factors_sycl.h>

#include "bundle_adjustment_sycl.h"
#include "bundle_adjustment_sycl_sparse.h"
#include "iterations.h"
#include "options.h"

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::benchmark::bundle_adjustment;
using namespace tinyopt::benchmark::bundle_adjustment_sycl;

TEST_CASE("SYCL BA factor accumulation", "[benchmark][sycl][bundle-adjustment]") {
  const Factor factor = MakeFactor();
  const Parameters initial_value = InitialValue();
  const Parameters target = GroundTruth();
  const auto objective = [&](const Parameters& value, auto& gradient) {
    std::array<double, Dims> parameters{};
    for (int i = 0; i < Dims; ++i) parameters[static_cast<std::size_t>(i)] = value[i];
    double cost = 0.0;
    constexpr Index CheckFactorCount = 6;
    for (Index index = 0; index < CheckFactorCount; ++index) {
      const Contribution contribution = factor(index, parameters);
      cost += contribution.cost;
      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        for (int i = 0; i < Dims; ++i)
          gradient[i] += contribution.gradient[static_cast<std::size_t>(i)];
      }
    }
    return cost;
  };
  REQUIRE(diff::CheckGradient(initial_value, objective, 1e-4, diff::Method::kCentral, false));

  Options options = CreateOptions();
  options.lm.jacobi_scaling = true;
  options.stop.min_rerr_dec = 1e-4;
  options.stop.min_step_norm2 = 0.0;
  options.stop.min_step_rel_norm2 = 1e-16;
  options.log.enable = false;
  options.exp.measure_time = false;
  options.exp.save_history = false;

  auto optimize_cpu = [&](std::uint16_t num_threads) {
    Parameters value = initial_value;
    Options run_options = options;
    run_options.num_threads = num_threads;
    const auto host_factor = [&](Index index, const Parameters& x, Parameters& gradient,
                                 Matrix<double, Dims, Dims>& hessian) {
      return AccumulateCpuFactor(factor, index, x, gradient, hessian);
    };
    const Summary result = OptimizeFactors(value, FactorCount, host_factor, run_options);
    REQUIRE(result.Succeeded());
    REQUIRE(result.Converged());
    REQUIRE((value - target).norm() < 1e-5);
    return std::pair{value, result};
  };

  sycl::queue queue{sycl::gpu_selector_v};
  REQUIRE(queue.get_device().is_gpu());
  Parameters sycl_value = initial_value;
  const Summary sycl_result = OptimizeFactors(queue, sycl_value, FactorCount, factor, options);
  REQUIRE(sycl_result.Succeeded());
  REQUIRE(sycl_result.Converged());
  REQUIRE((sycl_value - target).norm() < 1e-5);

  const auto [cpu_value, cpu_result] = optimize_cpu(1);
  REQUIRE((sycl_value - cpu_value).norm() < 1e-5);
  REQUIRE(sycl_result.final_cost.cost == Approx(cpu_result.final_cost.cost).margin(1e-6));
  PrintIterations("Bundle adjustment", "3-point fixed-size dense, " + std::to_string(FactorCount) +
                                           " factors",
                  "tinyopt-sycl", sycl_result.num_iters, sycl_result.Converged());

  BENCHMARK("CPU, 1 thread") { return optimize_cpu(1).second.final_cost.cost; };
#if defined(TINYOPT_USE_OPENMP)
  BENCHMARK("OpenMP, 8 threads") { return optimize_cpu(8).second.final_cost.cost; };
#endif
  BENCHMARK("AdaptiveCpp CUDA") {
    Parameters value = initial_value;
    const Summary result = OptimizeFactors(queue, value, FactorCount, factor, options);
    REQUIRE(result.Succeeded());
    REQUIRE(result.Converged());
    return result.final_cost.cost;
  };
}

TEST_CASE("SYCL large block-sparse BA accumulation",
          "[benchmark][sycl][bundle-adjustment][sparse]") {
  const Problem problem = MakeProblem(100, 50000);
  const BlockSparseLoss loss;
  const Index dims = 6 * problem.CameraCount() + 3 * problem.PointCount();
  Vector<double, Dynamic> cpu_gradient = Vector<double, Dynamic>::Zero(dims);
  Vector<double, Dynamic> sycl_gradient = Vector<double, Dynamic>::Zero(dims);
  BlockSparseLoss::Hessian cpu_hessian(problem.CameraCount(), problem.PointCount());
  BlockSparseLoss::Hessian sycl_hessian(problem.CameraCount(), problem.PointCount());
#if defined(TINYOPT_SYCL_TEST_GPU)
  sycl::queue queue{sycl::gpu_selector_v};
  REQUIRE(queue.get_device().is_gpu());
#else
  sycl::queue queue{sycl::default_selector_v};
#endif
  SyclBlockSparseAccumulator accumulator(queue, problem, sycl_hessian);

  cpu_hessian.setZero();
  const Cost cpu_cost = loss(problem, cpu_gradient, cpu_hessian);
  const Cost sycl_cost = accumulator.Accumulate(sycl_gradient);
  const double cost_tolerance = 1e-10 * std::max(1.0, std::abs(cpu_cost.cost));
  REQUIRE(sycl_cost.cost == Approx(cpu_cost.cost).margin(cost_tolerance));
  REQUIRE(sycl_cost.num_resisuals == cpu_cost.num_resisuals);
  REQUIRE((sycl_gradient - cpu_gradient).norm() < 1e-9 * std::max(1.0, cpu_gradient.norm()));

  auto maximum_block_error = [](const auto& lhs, const auto& rhs) {
    REQUIRE(lhs.size() == rhs.size());
    double error = 0.0;
    for (std::size_t block = 0; block < lhs.size(); ++block)
      error = std::max(error, (lhs[block] - rhs[block]).cwiseAbs().maxCoeff());
    return error;
  };
  const double hessian_scale = std::max(1.0, cpu_hessian.norm());
  REQUIRE(maximum_block_error(cpu_hessian.blocks<6, 6>(), sycl_hessian.blocks<6, 6>()) <
          1e-10 * hessian_scale);
  REQUIRE(maximum_block_error(cpu_hessian.blocks<3, 3>(), sycl_hessian.blocks<3, 3>()) <
          1e-10 * hessian_scale);
  REQUIRE(maximum_block_error(cpu_hessian.blocks<6, 3>(), sycl_hessian.blocks<6, 3>()) <
          1e-10 * hessian_scale);

  std::printf("SYCL block-sparse BA accumulation verified: %d cameras, %d points, %zu observations\n",
              problem.CameraCount(), problem.PointCount(), problem.observations.size());

  BENCHMARK("CPU serial block-sparse accumulation, 100c 50kp") {
    cpu_gradient.setZero();
    cpu_hessian.setZero();
    return loss(problem, cpu_gradient, cpu_hessian).cost;
  };
#if defined(TINYOPT_SYCL_TEST_GPU)
  BENCHMARK("AdaptiveCpp CUDA block-sparse accumulation, 100c 50kp") {
#else
  BENCHMARK("AdaptiveCpp device block-sparse accumulation, 100c 50kp") {
#endif
    return accumulator.Accumulate(sycl_gradient).cost;
  };
}
