// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include "problems.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimizers/lbfgs.h>
#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::ceres_exact;

namespace {

inline Options MakeTinyoptLBFGSOptions() {
  Options options(Options::Solver::LBFGS);
  options.log.enable = false;
  options.stop.max_iters = 50;
  options.stop.max_consec_failures = 20;
  options.stop.min_error = 0.0;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_grad_max = 1e-10;
  options.stop.min_grad_norm2 = 0.0;
  options.stop.min_step_norm2 = 0.0;
  options.stop.min_step_rel_norm2 = 1e-16;
  options.lbfgs.step_size = 1.0f;
  options.line_search.method = Options::LineSearchMethod::Wolfe;
  return options;
}

inline ceres::GradientProblemSolver::Options MakeCeresLBFGSOptions(const Options &tiny_opts) {
  ceres::GradientProblemSolver::Options ceres_opts;
  ceres_opts.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
  ceres_opts.line_search_type = ceres::LineSearchType::WOLFE;
  ceres_opts.max_lbfgs_rank = tiny_opts.lbfgs.history_size;
  ceres_opts.max_num_iterations = tiny_opts.stop.max_iters;
  ceres_opts.function_tolerance = tiny_opts.stop.min_rerr_dec;
  ceres_opts.gradient_tolerance = tiny_opts.stop.min_grad_max;
  ceres_opts.parameter_tolerance = std::sqrt(tiny_opts.stop.min_step_rel_norm2);
  ceres_opts.logging_type = ceres::SILENT;
  return ceres_opts;
}

}  // namespace

TEST_CASE("exact_lbfgs_nonlinear_system_4d", "[ceres_exact][lbfgs]") {
  const auto options = MakeTinyoptLBFGSOptions();
  const auto ceres_options = MakeCeresLBFGSOptions(options);

  // 1. Solve with Ceres L-BFGS
  auto ceres_x = NonlinearSystem4D::Initial();
  {
    ceres::GradientProblem problem(new NonlinearSystem4D::CeresFirstOrder());
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(ceres_options, problem, ceres_x.data(), &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt L-BFGS
    auto tinyopt_x = NonlinearSystem4D::Initial();
    lbfgs::Optimizer<NonlinearSystem4D::Vec> opt(options);
    const auto sum = opt(tinyopt_x, [](const auto &x, auto &g) {
      return NonlinearSystem4D::EvaluateObjective(x, g);
    });

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("NonlinearSystem4D - Ceres iters: " << ceres_iters
                                             << ", Tinyopt iters: " << tinyopt_iters);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());

    // Solution check (within tolerance)
    for (int i = 0; i < NonlinearSystem4D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-7));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-7);

    // Iterations comparison: with line search, L-BFGS matches Ceres within 3 iters
    INFO("Tinyopt vs Ceres iteration delta: " << (tinyopt_iters - ceres_iters));
    REQUIRE(tinyopt_iters <= ceres_iters + 3);
  }
}

TEST_CASE("exact_lbfgs_quartic_6d", "[ceres_exact][lbfgs]") {
  const auto options = MakeTinyoptLBFGSOptions();
  const auto ceres_options = MakeCeresLBFGSOptions(options);

  // 1. Solve with Ceres L-BFGS
  auto ceres_x = Quartic6D::Initial();
  {
    ceres::GradientProblem problem(new Quartic6D::CeresFirstOrder());
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(ceres_options, problem, ceres_x.data(), &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt L-BFGS
    auto tinyopt_x = Quartic6D::Initial();
    lbfgs::Optimizer<Quartic6D::Vec> opt(options);
    const auto sum =
        opt(tinyopt_x, [](const auto &x, auto &g) { return Quartic6D::EvaluateObjective(x, g); });

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("Quartic6D - Ceres iters: " << ceres_iters << ", Tinyopt iters: " << tinyopt_iters);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());

    // Solution check (within tolerance)
    for (int i = 0; i < Quartic6D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-8));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-8);

    // Iterations comparison
    INFO("Tinyopt vs Ceres iteration delta: " << (tinyopt_iters - ceres_iters));
    REQUIRE(tinyopt_iters <= ceres_iters + 3);
  }
}
