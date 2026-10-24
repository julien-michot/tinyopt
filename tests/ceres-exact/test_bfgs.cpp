// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include "problems.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimizers/bfgs.h>
#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::ceres_exact;

namespace {

inline Options MakeTinyoptBFGSOptions() {
  Options options(Options::Solver::BFGS);
  options.log.enable = false;
  options.stop.max_iters = 50;
  options.stop.max_consec_failures = 20;
  options.stop.min_error = 0.0;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_grad_max = 1e-10;
  options.stop.min_grad_norm2 = 0.0;
  options.stop.min_step_norm2 = 0.0;
  options.stop.min_step_rel_norm2 = 1e-16;
  options.bfgs.step_size = 1.0f;
  options.line_search.method = Options::LineSearchMethod::Wolfe;
  return options;
}

inline ceres::GradientProblemSolver::Options MakeCeresBFGSOptions(const Options &tiny_opts) {
  ceres::GradientProblemSolver::Options ceres_opts;
  ceres_opts.line_search_direction_type = ceres::LineSearchDirectionType::BFGS;
  ceres_opts.line_search_type = ceres::LineSearchType::WOLFE;
  ceres_opts.line_search_interpolation_type = ceres::LineSearchInterpolationType::BISECTION;
  ceres_opts.max_num_iterations = tiny_opts.stop.max_iters;
  ceres_opts.function_tolerance = tiny_opts.stop.min_rerr_dec;
  ceres_opts.gradient_tolerance = tiny_opts.stop.min_grad_max;
  ceres_opts.parameter_tolerance = std::sqrt(tiny_opts.stop.min_step_rel_norm2);
  ceres_opts.logging_type = ceres::SILENT;
  return ceres_opts;
}

}  // namespace

TEST_CASE("exact_bfgs_nonlinear_system_4d", "[ceres_exact][bfgs]") {
  const auto options = MakeTinyoptBFGSOptions();
  const auto ceres_options = MakeCeresBFGSOptions(options);

  // 1. Solve with Ceres BFGS
  auto ceres_x = NonlinearSystem4D::Initial();
  {
    ceres::GradientProblem problem(new NonlinearSystem4D::CeresFirstOrder());
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(ceres_options, problem, ceres_x.data(), &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt BFGS
    auto tinyopt_x = NonlinearSystem4D::Initial();
    bfgs::Optimizer<NonlinearSystem4D::Vec> opt(options);
    const auto sum = opt(tinyopt_x, [](const auto &x, auto &g) {
      return NonlinearSystem4D::EvaluateObjective(x, g);
    });

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("NonlinearSystem4D - Ceres iters: "
         << ceres_iters << " (cost evals: " << summary.num_cost_evaluations << ", grad evals: "
         << summary.num_gradient_evaluations << "), Tinyopt iters: " << tinyopt_iters
         << ", stop_reason: " << static_cast<int>(sum.stop_reason)
         << ", final_cost: " << sum.final_cost.cost);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
    // Solution check (machine precision match to Ceres)
    for (int i = 0; i < NonlinearSystem4D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-8));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-8);

    // Iterations comparison
    // Tinyopt outer trust-region step adaptation takes slightly more trial steps
    // than Ceres multi-evaluation inner line search (14 vs 10).
    INFO("Tinyopt vs Ceres iteration delta: " << (tinyopt_iters - ceres_iters));
    REQUIRE(tinyopt_iters <= ceres_iters + 5);
  }
}

TEST_CASE("exact_bfgs_quartic_6d", "[ceres_exact][bfgs]") {
  const auto options = MakeTinyoptBFGSOptions();
  const auto ceres_options = MakeCeresBFGSOptions(options);

  // 1. Solve with Ceres BFGS
  auto ceres_x = Quartic6D::Initial();
  {
    ceres::GradientProblem problem(new Quartic6D::CeresFirstOrder());
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(ceres_options, problem, ceres_x.data(), &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt BFGS
    auto tinyopt_x = Quartic6D::Initial();
    bfgs::Optimizer<Quartic6D::Vec> opt(options);
    const auto sum =
        opt(tinyopt_x, [](const auto &x, auto &g) { return Quartic6D::EvaluateObjective(x, g); });

    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("Quartic6D - Ceres iters: " << ceres_iters << ", Tinyopt iters: " << tinyopt_iters);
    // Solution check
    for (int i = 0; i < Quartic6D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-8));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-8);

    INFO("Tinyopt vs Ceres iteration delta: " << (tinyopt_iters - ceres_iters));
    REQUIRE(tinyopt_iters <= ceres_iters + 5);
  }
}

TEST_CASE("exact_bfgs_quadratic_4d", "[ceres_exact][bfgs]") {
  auto options = MakeTinyoptBFGSOptions();
  options.bfgs.step_size = 1.0f;
  const auto ceres_options = MakeCeresBFGSOptions(options);

  // 1. Solve with Ceres BFGS
  auto ceres_x = Quadratic4D::Initial();
  {
    ceres::GradientProblem problem(new Quadratic4D::CeresFirstOrder());
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(ceres_options, problem, ceres_x.data(), &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt BFGS
    auto tinyopt_x = Quadratic4D::Initial();
    bfgs::Optimizer<Quadratic4D::Vec> opt(options);
    const auto sum =
        opt(tinyopt_x, [](const auto &x, auto &g) { return Quadratic4D::EvaluateObjective(x, g); });

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("Quadratic4D - Ceres iters: " << ceres_iters << ", Tinyopt iters: " << tinyopt_iters);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());

    // Solution check
    for (int i = 0; i < Quadratic4D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-8));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-8);

    INFO("Tinyopt vs Ceres iteration delta: " << (tinyopt_iters - ceres_iters));
    REQUIRE(tinyopt_iters <= ceres_iters + 5);
  }
}
