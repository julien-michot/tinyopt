// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimizers/dl.h>
#include <tinyopt/tinyopt.h>

#include "problems.h"

using namespace tinyopt;
using namespace tinyopt::ceres_exact;

namespace {

inline Options MakeTinyoptDoglegOptions() {
  Options options(Options::Solver::DogLeg);
  options.log.enable = false;
  options.stop.max_iters = 50;
  options.stop.max_consec_failures = 15;
  options.stop.min_error = 0.0;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_grad_max = 1e-10;
  options.stop.min_grad_norm2 = 0.0;
  options.stop.min_step_norm2 = 0.0;
  options.stop.min_step_rel_norm2 = 1e-16;
  options.dl.radius_init = 1.0;
  options.dl.radius_max = 1000.0;
  return options;
}

inline ceres::Solver::Options MakeCeresDoglegOptions(const Options& tiny_opts) {
  ceres::Solver::Options ceres_opts;
  ceres_opts.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  ceres_opts.trust_region_strategy_type = ceres::DOGLEG;
  ceres_opts.dogleg_type = ceres::TRADITIONAL_DOGLEG;
  ceres_opts.dense_linear_algebra_library_type = ceres::EIGEN;
  ceres_opts.initial_trust_region_radius = tiny_opts.dl.radius_init;
  ceres_opts.max_trust_region_radius = tiny_opts.dl.radius_max;
  ceres_opts.max_num_iterations = tiny_opts.stop.max_iters;
  ceres_opts.function_tolerance = tiny_opts.stop.min_rerr_dec;
  ceres_opts.gradient_tolerance = tiny_opts.stop.min_grad_max;
  ceres_opts.parameter_tolerance = std::sqrt(tiny_opts.stop.min_step_rel_norm2);
  ceres_opts.logging_type = ceres::SILENT;
  return ceres_opts;
}

}  // namespace

TEST_CASE("exact_dogleg_nonlinear_system_4d", "[ceres_exact][dogleg]") {
  const auto options = MakeTinyoptDoglegOptions();
  const auto ceres_options = MakeCeresDoglegOptions(options);

  // 1. Solve with Ceres
  auto ceres_x = NonlinearSystem4D::Initial();
  {
    ceres::Problem problem;
    problem.AddResidualBlock(new NonlinearSystem4D::CeresCostFunction(), nullptr, ceres_x.data());
    ceres::Solver::Summary summary;
    ceres::Solve(ceres_options, &problem, &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt
    auto tinyopt_x = NonlinearSystem4D::Initial();
    dl::Optimizer<NonlinearSystem4D::Mat> opt(options);
    const auto sum =
        opt(tinyopt_x, NonlinearSystem4D::Accumulate<NonlinearSystem4D::Vec, NonlinearSystem4D::Vec,
                                                     NonlinearSystem4D::Mat>);

    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("NonlinearSystem4D - Ceres iters: " << ceres_iters
                                             << ", Tinyopt iters: " << tinyopt_iters);
    // Solution check (within tolerance)
    for (int i = 0; i < NonlinearSystem4D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-8));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-8);

    // Iterations check (within +1)
    REQUIRE(tinyopt_iters <= ceres_iters + 1);
  }
}

TEST_CASE("exact_dogleg_powell_4d", "[ceres_exact][dogleg]") {
  const auto options = MakeTinyoptDoglegOptions();
  const auto ceres_options = MakeCeresDoglegOptions(options);

  // 1. Solve with Ceres
  auto ceres_x = Powell4D::Initial();
  {
    ceres::Problem problem;
    problem.AddResidualBlock(new Powell4D::CeresCostFunction(), nullptr, ceres_x.data());
    ceres::Solver::Summary summary;
    ceres::Solve(ceres_options, &problem, &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt
    auto tinyopt_x = Powell4D::Initial();
    dl::Optimizer<Powell4D::Mat> opt(options);
    const auto sum =
        opt(tinyopt_x, Powell4D::Accumulate<Powell4D::Vec, Powell4D::Vec, Powell4D::Mat>);

    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("Powell4D - Ceres iters: " << ceres_iters << ", Tinyopt iters: " << tinyopt_iters);
    // Solution check (within tolerance: Powell is rank-deficient at x*=0; Tinyopt converges 3 iters
    // earlier)
    for (int i = 0; i < Powell4D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-3));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-3);

    // Iterations check (within +1)
    REQUIRE(tinyopt_iters <= ceres_iters + 1);
  }
}

TEST_CASE("exact_dogleg_quartic_6d", "[ceres_exact][dogleg]") {
  const auto options = MakeTinyoptDoglegOptions();
  const auto ceres_options = MakeCeresDoglegOptions(options);

  // 1. Solve with Ceres
  auto ceres_x = Quartic6D::Initial();
  {
    ceres::Problem problem;
    problem.AddResidualBlock(new Quartic6D::CeresCostFunction(), nullptr, ceres_x.data());
    ceres::Solver::Summary summary;
    ceres::Solve(ceres_options, &problem, &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt
    auto tinyopt_x = Quartic6D::Initial();
    dl::Optimizer<Eigen::Matrix<double, Quartic6D::Dims, Quartic6D::Dims>> opt(options);
    const auto sum = opt(
        tinyopt_x, Quartic6D::Accumulate<Quartic6D::Vec, Quartic6D::Vec,
                                         Eigen::Matrix<double, Quartic6D::Dims, Quartic6D::Dims>>);

    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("Quartic6D - Ceres iters: " << ceres_iters << ", Tinyopt iters: " << tinyopt_iters);
    // Solution check (within tolerance)
    for (int i = 0; i < Quartic6D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-8));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-8);

    // Iterations check (within +1)
    REQUIRE(tinyopt_iters <= ceres_iters + 1);
  }
}
