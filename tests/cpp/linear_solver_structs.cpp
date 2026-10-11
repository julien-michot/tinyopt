// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <tinyopt/math/linear_solvers.h>
#include <vector>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("DenseLinearSolvers solves dynamic and fixed size systems", "[solver][dense]") {
  const int n = 5;
  Eigen::Matrix<double, 5, 5> dense_matrix;
  dense_matrix << 6.0, 1.0, 0.5, 0.0, 0.0,
                  0.0, 5.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 4.0, 0.5, 0.0,
                  0.0, 0.0, 0.0, 3.0, 0.25,
                  0.0, 0.0, 0.0, 0.0, 2.0;

  Eigen::Matrix<double, 5, 5> full_matrix = dense_matrix.selfadjointView<Eigen::Upper>();
  VecX expected(n);
  expected << 1.0, -2.0, 0.5, 3.0, -1.0;
  const VecX rhs = full_matrix * expected;

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
  methods.push_back(LinearSolverMethod::SVD);
#endif

  SECTION("dynamic size MatX") {
    MatX A = dense_matrix;
    if (RequiresFullMatrix(LinearSolverMethod::LU)) {
      CompleteSymmetricMatrix(A);
    }
    for (auto method : methods) {
      DenseLinearSolvers<MatX> solver(method, n, LinearSolverOptions{.svd_relative_threshold = 0.0});
      REQUIRE(solver.GetMethod() == method);
      REQUIRE(solver.Dims() == n);

      solver.resize(n);
      solver.analyze(A);

      MatX A_mat = dense_matrix;
      if (RequiresFullMatrix(method)) {
        CompleteSymmetricMatrix(A_mat);
      }
      auto sol = solver.solve(A_mat, rhs);
      REQUIRE(sol.has_value());
      REQUIRE((sol.value() - expected).norm() < 1e-10);
    }
  }

  SECTION("fixed size Matrix5d") {
    using FixedMat = Eigen::Matrix<double, 5, 5>;
    using FixedVec = Vector<double, 5>;
    FixedVec fixed_expected = expected;
    FixedVec fixed_rhs = rhs;

    for (auto method : methods) {
      DenseLinearSolvers<FixedMat> solver(method);
      REQUIRE(solver.GetMethod() == method);

      solver.resize(5);
      solver.analyze(dense_matrix);

      FixedMat A_mat = dense_matrix;
      if (RequiresFullMatrix(method)) {
        CompleteSymmetricMatrix(A_mat);
      }
      auto sol = solver.solve(A_mat, fixed_rhs);
      REQUIRE(sol.has_value());
      REQUIRE((sol.value() - fixed_expected).norm() < 1e-10);
    }
  }

  SECTION("detects indefinite or singular matrix") {
    MatX singular = MatX::Zero(3, 3);
    singular(0, 0) = 1.0;
    DenseLinearSolvers<MatX> solver(LinearSolverMethod::LDLT, 3);
    auto sol = solver.solve(singular, Vec3::Ones());
    REQUIRE_FALSE(sol.has_value());
  }
}

TEST_CASE("SparseLinearSolvers solves sparse systems", "[solver][sparse]") {
  const int n = 5;
  Eigen::Matrix<double, 5, 5> dense_matrix;
  dense_matrix << 6.0, 1.0, 0.5, 0.0, 0.0,
                  0.0, 5.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 4.0, 0.5, 0.0,
                  0.0, 0.0, 0.0, 3.0, 0.25,
                  0.0, 0.0, 0.0, 0.0, 2.0;

  SparseMat sparse_upper = dense_matrix.sparseView();
  Eigen::Matrix<double, 5, 5> full_matrix = dense_matrix.selfadjointView<Eigen::Upper>();
  VecX expected(n);
  expected << 1.0, -2.0, 0.5, 3.0, -1.0;
  const VecX rhs = full_matrix * expected;

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
#if defined(TINYOPT_ENABLE_SUITESPARSE)
  methods.push_back(LinearSolverMethod::SuiteSparse);
#endif

  for (auto method : methods) {
    SparseLinearSolvers<SparseMat> solver(method, n);
    REQUIRE(solver.GetMethod() == method);

    SparseMat A = sparse_upper;
    if (RequiresFullMatrix(method)) {
      CompleteSymmetricMatrix(A);
    }
#if defined(TINYOPT_ENABLE_SUITESPARSE)
    if (method == LinearSolverMethod::SuiteSparse) {
      A.makeCompressed();
    }
#endif

    solver.analyze(A);
    if (method != LinearSolverMethod::SuiteSparse) {
      REQUIRE(solver.IsAnalyzed());
    }

    auto sol = solver.solve(A, rhs);
    REQUIRE(sol.has_value());
    REQUIRE((sol.value() - expected).norm() < 1e-10);
  }

  SECTION("detects singular sparse matrix") {
    SparseMat singular(3, 3);
    singular.insert(0, 0) = 1.0;
    SparseLinearSolvers<SparseMat> solver(LinearSolverMethod::LDLT, 3);
    auto sol = solver.solve(singular, Vec3::Ones());
    REQUIRE_FALSE(sol.has_value());
  }
}
