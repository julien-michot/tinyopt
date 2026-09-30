// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/solvers/gn.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::solvers;

TEMPLATE_TEST_CASE("tinyopt_solver_gn_numdiff", "[solver]", SolverGN<Mat2>, SolverGN<MatX>) {
  TestType solver;
  using Vec = typename TestType::Grad_t;
  Vec x = Vec::Zero(2);
  const Vec2 target(4, 5);
  const auto residuals = [&](const auto &value) { return (value - target).eval(); };

  REQUIRE(solver.Build(x, diff::CreateNumDiffFunc2(x, residuals)));
  const auto maybe_dx = solver.Solve();
  REQUIRE(maybe_dx.has_value());
  REQUIRE((*maybe_dx - target).norm() < 1e-2);
}