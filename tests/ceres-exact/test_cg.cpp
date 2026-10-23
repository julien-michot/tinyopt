// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/optimizers/cg.h>
#include <tinyopt/tinyopt.h>

#include "problems.h"

using namespace tinyopt;
using namespace tinyopt::ceres_exact;

namespace {

inline Options MakeTinyoptCGOptions() {
  Options options(Options::Solver::ConjugateGradient);
  options.log.enable = false;
  options.stop.max_iters = 100;
  options.stop.max_consec_failures = 20;
  options.stop.min_error = 0.0;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_grad_max = 1e-10;
  options.stop.min_grad_norm2 = 0.0;
  options.stop.min_step_norm2 = 0.0;
  options.stop.min_step_rel_norm2 = 1e-16;
  options.cg.step_size = 0.5f;
  return options;
}

inline ceres::GradientProblemSolver::Options MakeCeresCGOptions(const Options& tiny_opts) {
  ceres::GradientProblemSolver::Options ceres_opts;
  ceres_opts.line_search_direction_type =
      ceres::LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT;
  ceres_opts.nonlinear_conjugate_gradient_type =
      ceres::NonlinearConjugateGradientType::POLAK_RIBIERE;
  ceres_opts.line_search_type = ceres::LineSearchType::WOLFE;
  ceres_opts.max_num_iterations = tiny_opts.stop.max_iters;
  ceres_opts.function_tolerance = tiny_opts.stop.min_rerr_dec;
  ceres_opts.gradient_tolerance = tiny_opts.stop.min_grad_max;
  ceres_opts.parameter_tolerance = 1e-8;
  ceres_opts.logging_type = ceres::SILENT;
  return ceres_opts;
}

}  // namespace

TEST_CASE("exact_cg_quartic_6d", "[ceres_exact][cg]") {
  const auto options = MakeTinyoptCGOptions();
  const auto ceres_options = MakeCeresCGOptions(options);

  // 1. Solve with Ceres CG
  auto ceres_x = Quartic6D::Initial();
  {
    ceres::GradientProblem problem(new Quartic6D::CeresFirstOrder());
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(ceres_options, problem, ceres_x.data(), &summary);
    REQUIRE(summary.termination_type == ceres::CONVERGENCE);

    // 2. Solve with Tinyopt CG
    auto tinyopt_x = Quartic6D::Initial();
    cg::Optimizer<Quartic6D::Vec> opt(options);
    const auto sum = opt(tinyopt_x, Quartic6D::EvaluateObjective<Quartic6D::Vec, Quartic6D::Vec>);

    const int ceres_iters = static_cast<int>(summary.iterations.size()) - 1;
    const int tinyopt_iters = sum.num_iters;

    INFO("Quartic6D - Ceres iters: " << ceres_iters << ", Tinyopt iters: " << tinyopt_iters);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());

    // Solution check (within tolerance)
    for (int i = 0; i < Quartic6D::Dims; ++i) {
      REQUIRE(tinyopt_x[i] == Catch::Approx(ceres_x[i]).margin(1e-4));
    }
    REQUIRE((tinyopt_x - ceres_x).norm() < 1e-4);

    // Check iterations
    INFO("Tinyopt vs Ceres iteration delta: " << (tinyopt_iters - ceres_iters));
    REQUIRE(tinyopt_iters <= ceres_iters + 1);
  }
}
