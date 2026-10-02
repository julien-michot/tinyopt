// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <vector>
#include <tinyopt/solvers/gn.h>

using Catch::Approx;
using namespace tinyopt;

template <typename Hessian>
void CheckLinearSolverMethods() {
  using Solver = solvers::SolverGN<Hessian>;
  using VectorType = typename Solver::Grad_t;

  Mat3 dense_matrix;
  dense_matrix << 6.0, 1.0, 0.5, 1.0, 5.0, 1.0, 0.5, 1.0, 4.0;
  Hessian matrix;
  if constexpr (traits::is_sparse_matrix_v<Hessian>)
    matrix = dense_matrix.sparseView();
  else
    matrix = dense_matrix;

  VectorType expected(3);
  expected << 1.0, -2.0, 0.5;
  const VectorType rhs = matrix * expected;
  const VectorType initial = VectorType::Zero(3);
  std::vector<LinearSolverMethod> methods;
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
  methods.push_back(LinearSolverMethod::LDLT);
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
  methods.push_back(LinearSolverMethod::LLT);
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
  methods.push_back(LinearSolverMethod::LU);
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
  methods.push_back(LinearSolverMethod::QR);
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
  if constexpr (!traits::is_sparse_matrix_v<Hessian>) methods.push_back(LinearSolverMethod::SVD);
#endif

  auto check_method = [&](LinearSolverMethod method) {
    Options options;
    options.linear_solver = method;
    options.log.enable = false;
    Solver solver(options);
    auto residual = [&](const auto &x, auto &gradient, auto &hessian) {
      const auto values = (matrix * x - rhs).eval();
      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        gradient.noalias() = matrix.transpose() * values;
        hessian = matrix.transpose() * matrix;
      }
      return Cost(values.norm(), values.size());
    };
    REQUIRE(solver.Build(initial, residual));
    const auto step = solver.Solve();
    REQUIRE(step.has_value());
    REQUIRE((step.value() - expected).norm() < 1e-8);
  };

  for (const auto method : methods) check_method(method);
}

TEST_CASE("linear solver methods solve the same dense and sparse problem", "[solver]") {
  SECTION("dense") { CheckLinearSolverMethods<Mat3>(); }
  SECTION("sparse") { CheckLinearSolverMethods<SparseMat>(); }
}