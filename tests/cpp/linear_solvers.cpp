// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/solvers/gn.h>
#include <vector>

using Catch::Approx;
using namespace tinyopt;

template <typename Hessian>
void CheckLinearSolverMethods() {
  using Solver = solvers::SolverGN<Hessian>;
  using VectorType = typename Solver::Grad_t;

  Eigen::Matrix<double, 5, 5> dense_matrix;
  dense_matrix << 6.0, 1.0, 0.5, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.5, 1.0, 4.0, 0.5, 0.0, 0.0,
      0.0, 0.5, 3.0, 0.25, 0.0, 0.0, 0.0, 0.25, 2.0;
  Hessian matrix;
  if constexpr (traits::is_sparse_matrix_v<Hessian>)
    matrix = dense_matrix.sparseView();
  else
    matrix = dense_matrix;

  VectorType expected(5);
  expected << 1.0, -2.0, 0.5, 3.0, -1.0;
  const VectorType rhs = matrix * expected;
  const VectorType initial = VectorType::Zero(5);
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
  if constexpr (!traits::is_sparse_matrix_v<Hessian>) {
    methods.push_back(LinearSolverMethod::SVD);
  }
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
  SECTION("dense") { CheckLinearSolverMethods<Eigen::Matrix<double, 5, 5>>(); }
  SECTION("sparse") { CheckLinearSolverMethods<SparseMat>(); }
}

#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
TEST_CASE("SVD drops directions below the relative threshold", "[solver]") {
  MatX jacobian = MatX::Zero(3, 3);
  jacobian.diagonal() << 2.0, 1.0, 1e-4;
  VecX residual(3);
  residual << -4.0, -2.0, -1e-4;

  Options default_options;
  default_options.linear_solver = LinearSolverMethod::SVD;
  default_options.log.enable = false;
  solvers::SolverGN<MatX> default_solver(default_options);

  const auto accumulate = [&](const auto &, auto &gradient, auto &hessian) {
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.noalias() = jacobian.transpose() * residual;
      hessian.noalias() = jacobian.transpose() * jacobian;
    }
    return Cost(residual.norm(), residual.size());
  };

  REQUIRE(default_solver.Build(VecX::Zero(3), accumulate));
  const auto default_step = default_solver.Solve();
  REQUIRE(default_step.has_value());
  VecX default_expected(3);
  default_expected << 2.0, 2.0, 1.0;
  REQUIRE((*default_step - default_expected).norm() < 1e-10);

  Options options;
  options.linear_solver = LinearSolverMethod::SVD;
  options.svd_relative_threshold = 1e-6;
  options.log.enable = false;
  solvers::SolverGN<MatX> solver(options);

  REQUIRE(solver.Build(VecX::Zero(3), accumulate));
  const auto step = solver.Solve();
  REQUIRE(step.has_value());
  VecX expected(3);
  expected << 2.0, 2.0, 0.0;
  REQUIRE((*step - expected).norm() < 1e-10);
}
#endif

TEST_CASE("1x1 dense solve checks epsilon", "[solver]") {
  Eigen::Matrix<double, 1, 1> matrix;
  matrix(0, 0) = 2.0;
  Eigen::Matrix<double, 1, 1> rhs;
  rhs(0) = 6.0;
  const auto solution = SolveLinearSystem(matrix, rhs, LinearSolverMethod::LU);
  REQUIRE(solution.has_value());
  REQUIRE((*solution)(0) == Approx(3.0));

  matrix(0, 0) = Eigen::NumTraits<double>::epsilon();
  REQUIRE_FALSE(SolveLinearSystem(matrix, rhs, LinearSolverMethod::LU).has_value());
}

TEST_CASE("2x2 dense solve uses the upper triangle", "[solver]") {
  Mat2 matrix;
  matrix << 4.0, 1.0, 0.0, 3.0;
  const Vec2 expected(2.0, -1.0);
  Mat2 symmetric;
  symmetric << 4.0, 1.0, 1.0, 3.0;
  const Vec2 rhs = symmetric * expected;
  const auto solution = SolveLinearSystem(matrix, rhs, LinearSolverMethod::LDLT);
  REQUIRE(solution.has_value());
  REQUIRE((*solution - expected).norm() < 1e-12);
  const auto full_method_solution = SolveLinearSystem(matrix, rhs, LinearSolverMethod::LU);
  REQUIRE(full_method_solution.has_value());
  REQUIRE((*full_method_solution - expected).norm() < 1e-12);

  const double scale = 1e-100;
  const auto scaled_solution =
      SolveLinearSystem(symmetric * scale, rhs * scale, LinearSolverMethod::LDLT);
  REQUIRE(scaled_solution.has_value());
  REQUIRE((*scaled_solution - expected).norm() < 1e-12);

  Mat2 indefinite;
  indefinite << 2.0, 0.0, 0.0, -0.5;
  const Vec2 indefinite_rhs = indefinite * expected;
  const auto indefinite_solution =
      SolveLinearSystem(indefinite, indefinite_rhs, LinearSolverMethod::LU);
  REQUIRE(indefinite_solution.has_value());
  REQUIRE((*indefinite_solution - expected).norm() < 1e-12);

  Mat2 small_trace;
  small_trace << 1.0, 1e16, 0.0, -1.0 + 1e-12;
  REQUIRE_FALSE(SolveLinearSystem(small_trace, Vec2::Ones(), LinearSolverMethod::LDLT).has_value());

  matrix(1, 1) = Eigen::NumTraits<double>::epsilon() * 0.5;
  matrix(0, 0) = 1.0;
  matrix(0, 1) = 0.0;
  REQUIRE_FALSE(SolveLinearSystem(matrix, Vec2::Ones(), LinearSolverMethod::LDLT).has_value());
}

TEST_CASE("3x3 dense solve uses the upper triangle", "[solver]") {
  Mat3 matrix;
  matrix << 6.0, 1.0, 0.5, 0.0, 5.0, 1.0, 0.0, 0.0, 4.0;
  const Vec3 expected(1.0, -2.0, 0.5);
  Mat3 symmetric;
  symmetric << 6.0, 1.0, 0.5, 1.0, 5.0, 1.0, 0.5, 1.0, 4.0;
  const Vec3 rhs = symmetric * expected;
  const auto solution = SolveLinearSystem(matrix, rhs, LinearSolverMethod::LDLT);
  REQUIRE(solution.has_value());
  REQUIRE((*solution - expected).norm() < 1e-12);
  const auto qr_solution = SolveLinearSystem(matrix, rhs, LinearSolverMethod::QR);
  REQUIRE(qr_solution.has_value());
  REQUIRE((*qr_solution - expected).norm() < 1e-12);

  matrix.setZero();
  matrix(0, 0) = 1.0;
  matrix(1, 1) = 1.0;
  matrix(2, 2) = Eigen::NumTraits<double>::epsilon() * 0.5;
  REQUIRE_FALSE(SolveLinearSystem(matrix, Vec3::Ones(), LinearSolverMethod::LDLT).has_value());
}

TEST_CASE("4x4 dense solve uses the upper triangle", "[solver]") {
  Mat4 matrix;
  matrix << 8.0, 1.0, 0.5, 0.25, 0.0, 7.0, 1.0, 0.5, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 0.0, 5.0;
  Mat4 symmetric;
  symmetric << 8.0, 1.0, 0.5, 0.25, 1.0, 7.0, 1.0, 0.5, 0.5, 1.0, 6.0, 1.0, 0.25, 0.5, 1.0, 5.0;
  const Vec4 expected(1.0, -2.0, 0.5, 3.0);
  const Vec4 rhs = symmetric * expected;
  const auto solution = SolveLinearSystem(matrix, rhs, LinearSolverMethod::LDLT);
  REQUIRE(solution.has_value());
  REQUIRE((*solution - expected).norm() < 1e-12);

  Eigen::Matrix<float, 4, 4> scaled = Eigen::Matrix<float, 4, 4>::Zero();
  scaled.diagonal() << 1.0f, 1.0f, 1.0f, 20000.0f;
  const Eigen::Matrix<float, 4, 1> scaled_expected(1.0f, -2.0f, 0.5f, 3.0f);
  const auto scaled_solution =
      SolveLinearSystem(scaled, scaled * scaled_expected, LinearSolverMethod::LDLT);
  REQUIRE(scaled_solution.has_value());
  REQUIRE((*scaled_solution - scaled_expected).norm() < 1e-5f);

  Mat4 singular = Mat4::Identity();
  singular(3, 3) = Eigen::NumTraits<double>::epsilon() * Eigen::NumTraits<double>::epsilon() * 0.5;
  REQUIRE_FALSE(SolveLinearSystem(singular, Vec4::Ones(), LinearSolverMethod::LDLT).has_value());
}
