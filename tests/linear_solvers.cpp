// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/generators/catch_generators.hpp>
#include "tinyopt/solvers/options.h"
#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/solvers/solvers.h>

using Catch::Approx;

using namespace tinyopt;
using namespace tinyopt::solvers;

#if TINYOPT_BUILD_SOLVER_QR
TEST_CASE("tinyopt_solvers2_qr", "[solver]") {
  using SolverType = SolverLM<Mat2>;
  SolverType::Options options;
  options.linear_solver = solvers::Options2::LinearSolver::QR;
  SolverType solver(options);
  using Vec = typename SolverLM<Mat2>::Grad_t;
  SECTION("Resize") { solver.resize(2); }
  SECTION("Solve") {
    Vec x = Vec::Zero(2);
    const Vec2 y = Vec2(4, 5);

    int num_grad_updates = 0;
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      auto res = (x - y).eval();
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        grad = res;
        H = Mat2::Identity();
        num_grad_updates++;
      }
      return res;
    };

    bool built = solver.Build(x, loss);
    REQUIRE(built);
    REQUIRE(num_grad_updates == 1);

    solver.Rebuild(false);
    built = solver.Build(x, loss);
    REQUIRE(built);
    REQUIRE(num_grad_updates == 1);

    const auto &maybe_dx = solver.Solve();
    REQUIRE(maybe_dx.has_value());
    const auto &dx = maybe_dx.value();

    REQUIRE(dx[0] == Approx(y[0]).margin(1e-2));
    REQUIRE(dx[1] == Approx(y[1]).margin(1e-2));
  }
}
#endif


#if TINYOPT_BUILD_SOLVER_LU
TEST_CASE("tinyopt_solvers2_lu", "[solver]") {
  using SolverType = SolverLM<Mat2>;
  SolverType::Options options;
  options.linear_solver = solvers::Options2::LinearSolver::LU;
  SolverType solver(options);
  using Vec = typename SolverLM<Mat2>::Grad_t;
  SECTION("Resize") { solver.resize(2); }
  SECTION("Solve") {
    Vec x = Vec::Zero(2);
    const Vec2 y = Vec2(4, 5);

    int num_grad_updates = 0;
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      auto res = (x - y).eval();
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        grad = res;
        H = Mat2::Identity();
        num_grad_updates++;
      }
      return res;
    };

    bool built = solver.Build(x, loss);
    REQUIRE(built);
    REQUIRE(num_grad_updates == 1);

    solver.Rebuild(false);
    built = solver.Build(x, loss);
    REQUIRE(built);
    REQUIRE(num_grad_updates == 1);

    const auto &maybe_dx = solver.Solve();
    REQUIRE(maybe_dx.has_value());
    const auto &dx = maybe_dx.value();

    REQUIRE(dx[0] == Approx(y[0]).margin(1e-2));
    REQUIRE(dx[1] == Approx(y[1]).margin(1e-2));
  }
}
#endif


#if TINYOPT_BUILD_SOLVER_SVD
TEST_CASE("tinyopt_solvers2_svd", "[solver]") {
  using SolverType = SolverLM<Mat2>;
  SolverType::Options options;
  options.linear_solver = solvers::Options2::LinearSolver::SVD;
  SolverType solver(options);
  using Vec = typename SolverLM<Mat2>::Grad_t;
  SECTION("Resize") { solver.resize(2); }
  SECTION("Solve") {
    Vec x = Vec::Zero(2);
    const Vec2 y = Vec2(4, 5);

    int num_grad_updates = 0;
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      auto res = (x - y).eval();
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        grad = res;
        H = Mat2::Identity();
        num_grad_updates++;
      }
      return res;
    };

    bool built = solver.Build(x, loss);
    REQUIRE(built);
    REQUIRE(num_grad_updates == 1);

    solver.Rebuild(false);
    built = solver.Build(x, loss);
    REQUIRE(built);
    REQUIRE(num_grad_updates == 1);

    const auto &maybe_dx = solver.Solve();
    REQUIRE(maybe_dx.has_value());
    const auto &dx = maybe_dx.value();

    REQUIRE(dx[0] == Approx(y[0]).margin(1e-2));
    REQUIRE(dx[1] == Approx(y[1]).margin(1e-2));
  }
}
#endif
