// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <tinyopt/solvers/gn.h>

using namespace tinyopt;

TEST_CASE("SuiteSparse solves compressed and uncompressed sparse systems", "[solver][suitesparse]") {
  const Vec3 expected(1.0, -2.0, 0.5);

  SECTION("compressed upper triangle") {
    SparseMat hessian(3, 3);
    hessian.insert(0, 0) = 6.0;
    hessian.insert(0, 1) = 1.0;
    hessian.insert(0, 2) = 0.5;
    hessian.insert(1, 1) = 5.0;
    hessian.insert(1, 2) = 1.0;
    hessian.insert(2, 2) = 4.0;
    hessian.makeCompressed();

    const Vec3 rhs = hessian.selfadjointView<Eigen::Upper>() * expected;
    const auto solution = SolveLinearSystem(hessian, rhs, LinearSolverMethod::SuiteSparse);

    REQUIRE(solution.has_value());
    REQUIRE((solution.value() - expected).norm() < 1e-10);
  }

  SECTION("uncompressed matrix fallback") {
    SparseMat hessian(3, 3);
    hessian.coeffRef(0, 0) = 6.0;
    hessian.coeffRef(0, 1) = 1.0;
    hessian.coeffRef(0, 2) = 0.5;
    hessian.coeffRef(1, 1) = 5.0;
    hessian.coeffRef(1, 2) = 1.0;
    hessian.coeffRef(2, 2) = 4.0;
    REQUIRE_FALSE(hessian.isCompressed());

    const Vec3 rhs = hessian.selfadjointView<Eigen::Upper>() * expected;
    const auto solution = SolveLinearSystem(hessian, rhs, LinearSolverMethod::SuiteSparse);

    REQUIRE(solution.has_value());
    REQUIRE((solution.value() - expected).norm() < 1e-10);
  }
}

TEST_CASE("Gauss-Newton compresses its Hessian for SuiteSparse", "[solver][suitesparse]") {
  Options options;
  options.linear_solver = LinearSolverMethod::SuiteSparse;
  options.log.enable = false;
  solvers::SolverGN<SparseMat> solver(options);

  const Vec3 expected(1.0, -2.0, 0.5);
  const Vec3 initial = Vec3::Zero();
  auto accumulate = [&](const auto &x, auto &gradient, auto &hessian) {
    const Vec3 residual = x - expected;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient = residual;
      hessian.coeffRef(0, 0) = 1.0;
      hessian.coeffRef(1, 1) = 1.0;
      hessian.coeffRef(2, 2) = 1.0;
    }
    return Cost(residual.norm(), residual.size());
  };

  REQUIRE(solver.Build(initial, accumulate));
  REQUIRE(solver.Hessian().isCompressed());
  const auto step = solver.Solve();
  REQUIRE(step.has_value());
  REQUIRE((step.value() - expected).norm() < 1e-10);
}