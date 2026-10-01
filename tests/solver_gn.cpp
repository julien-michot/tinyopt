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

TEST_CASE("tinyopt_solver_gn_solves_normal_equation") {
  Options options;
  SolverGN<Mat2> solver(options);

  const Vec2 target(3.0, -2.0);
  const Mat2 H = (Mat2() << 4.0, 1.0, 1.0, 3.0).finished();
  const auto accumulation = [&](const auto &value, auto &gradient, auto &hessian) {
    const Vec2 residual = value - target;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient = H * residual;
      hessian = H;
    }
    return 0.5 * residual.dot(H * residual);
  };

  Vec2 x = Vec2::Zero();
  REQUIRE(solver.Build(x, accumulation));
  const auto step = solver.Solve();
  REQUIRE(step.has_value());
  REQUIRE(step->isApprox(target, 1e-6));
}