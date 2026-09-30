// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/solvers/bfgs.h>

using namespace tinyopt;
using namespace tinyopt::solvers;

template <typename Solver>
void CheckQuasiNewtonInitialStep() {
  Options options;
  options.bfgs.step_size = 0.1f;
  options.lbfgs.step_size = 0.1f;
  Solver solver(options);
  const Vec2 x = Vec2::Zero();
  const Vec2 target(4.0, 5.0);
  const auto accumulation = [&](const auto &, auto &gradient) {
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = -target;
    return Cost(1.0, 2);
  };

  REQUIRE(solver.Build(x, accumulation));
  const auto maybe_step = solver.Solve();
  REQUIRE(maybe_step.has_value());
  REQUIRE((*maybe_step - options.bfgs.step_size * target).norm() < 1e-5);
}

TEST_CASE("tinyopt_solver_bfgs_initial_step") { CheckQuasiNewtonInitialStep<SolverBFGS<Vec2>>(); }

TEST_CASE("tinyopt_solver_lbfgs_initial_step") {
  CheckQuasiNewtonInitialStep<SolverLBFGS<Vec2>>();
}