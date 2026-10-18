// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/math/solvers/block_sparse.h>
#include <tinyopt/math/solvers/dense.h>
#include <tinyopt/math/solvers/sparse.h>
#include <tinyopt/optimizers/gn.h>
#include <vector>

using Catch::Approx;
using namespace tinyopt;

template <typename Hessian>
void CheckLinearSolverMethods() {
  using VectorType = Vector<typename Hessian::Scalar, SQRT(traits::params_trait<Hessian>::Dims)>;

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
    options.solver_type = Options::Solver::GaussNewton;
    options.linear_solver = method;
    options.log.enable = false;
    gn::Optimizer<Hessian> optimizer(options);
    auto residual = [&](const auto &x, auto &gradient, auto &hessian) {
      const auto values = (matrix * x - rhs).eval();
      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        gradient.noalias() = matrix.transpose() * values;
        hessian = matrix.transpose() * matrix;
      }
      return Cost(values.norm(), values.size());
    };
    REQUIRE(optimizer.Build(initial, residual));
    const auto step = optimizer.SolveGN();
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
  default_options.solver_type = Options::Solver::GaussNewton;
  default_options.linear_solver = LinearSolverMethod::SVD;
  default_options.log.enable = false;
  gn::Optimizer<MatX> default_optimizer(default_options);

  const auto accumulate = [&](const auto &, auto &gradient, auto &hessian) {
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.noalias() = jacobian.transpose() * residual;
      hessian.noalias() = jacobian.transpose() * jacobian;
    }
    return Cost(residual.norm(), residual.size());
  };

  REQUIRE(default_optimizer.Build(VecX::Zero(3), accumulate));
  const auto default_step = default_optimizer.SolveGN();
  REQUIRE(default_step.has_value());
  VecX default_expected(3);
  default_expected << 2.0, 2.0, 1.0;
  REQUIRE((*default_step - default_expected).norm() < 1e-10);

  Options options;
  options.solver_type = Options::Solver::GaussNewton;
  options.linear_solver = LinearSolverMethod::SVD;
  options.svd_relative_threshold = 1e-6;
  options.log.enable = false;
  gn::Optimizer<MatX> optimizer(options);

  REQUIRE(optimizer.Build(VecX::Zero(3), accumulate));
  const auto step = optimizer.SolveGN();
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
  DenseLinearSolvers<Eigen::Matrix<double, 1, 1>> solver_lu(LinearSolverMethod::LU);
  const auto solution = solver_lu.Solve(matrix, rhs);
  REQUIRE(solution.has_value());
  REQUIRE((*solution)(0) == Approx(3.0));

  matrix(0, 0) = Eigen::NumTraits<double>::epsilon();
  REQUIRE_FALSE(solver_lu.Solve(matrix, rhs).has_value());
}

TEST_CASE("2x2 dense solve uses the upper triangle", "[solver]") {
  Mat2 matrix;
  matrix << 4.0, 1.0, 0.0, 3.0;
  const Vec2 expected(2.0, -1.0);
  Mat2 symmetric;
  symmetric << 4.0, 1.0, 1.0, 3.0;
  const Vec2 rhs = symmetric * expected;
  DenseLinearSolvers<Mat2> solver_ldlt(LinearSolverMethod::LDLT);
  const auto solution = solver_ldlt.Solve(matrix, rhs);
  REQUIRE(solution.has_value());
  REQUIRE((*solution - expected).norm() < 1e-12);
  DenseLinearSolvers<Mat2> solver_lu(LinearSolverMethod::LU);
  const auto full_method_solution = solver_lu.Solve(matrix, rhs);
  REQUIRE(full_method_solution.has_value());
  REQUIRE((*full_method_solution - expected).norm() < 1e-12);

  const double scale = 1e-100;
  const auto scaled_solution = solver_ldlt.Solve(symmetric * scale, rhs * scale);
  REQUIRE(scaled_solution.has_value());
  REQUIRE((*scaled_solution - expected).norm() < 1e-12);

  Mat2 indefinite;
  indefinite << 2.0, 0.0, 0.0, -0.5;
  const Vec2 indefinite_rhs = indefinite * expected;
  const auto indefinite_solution = solver_lu.Solve(indefinite, indefinite_rhs);
  REQUIRE(indefinite_solution.has_value());
  REQUIRE((*indefinite_solution - expected).norm() < 1e-12);

  Mat2 small_trace;
  small_trace << 1.0, 1e16, 0.0, -1.0 + 1e-12;
  REQUIRE_FALSE(solver_ldlt.Solve(small_trace, Vec2::Ones()).has_value());

  matrix(1, 1) = Eigen::NumTraits<double>::epsilon() * 0.5;
  matrix(0, 0) = 1.0;
  matrix(0, 1) = 0.0;
  REQUIRE_FALSE(solver_ldlt.Solve(matrix, Vec2::Ones()).has_value());
}

TEST_CASE("3x3 dense solve uses the upper triangle", "[solver]") {
  Mat3 matrix;
  matrix << 6.0, 1.0, 0.5, 0.0, 5.0, 1.0, 0.0, 0.0, 4.0;
  const Vec3 expected(1.0, -2.0, 0.5);
  Mat3 symmetric;
  symmetric << 6.0, 1.0, 0.5, 1.0, 5.0, 1.0, 0.5, 1.0, 4.0;
  const Vec3 rhs = symmetric * expected;
  DenseLinearSolvers<Mat3> solver_ldlt(LinearSolverMethod::LDLT);
  const auto solution = solver_ldlt.Solve(matrix, rhs);
  REQUIRE(solution.has_value());
  REQUIRE((*solution - expected).norm() < 1e-12);
  DenseLinearSolvers<Mat3> solver_qr(LinearSolverMethod::QR);
  const auto qr_solution = solver_qr.Solve(matrix, rhs);
  REQUIRE(qr_solution.has_value());
  REQUIRE((*qr_solution - expected).norm() < 1e-12);

  matrix.setZero();
  matrix(0, 0) = 1.0;
  matrix(1, 1) = 1.0;
  matrix(2, 2) = Eigen::NumTraits<double>::epsilon() * 0.5;
  REQUIRE_FALSE(solver_ldlt.Solve(matrix, Vec3::Ones()).has_value());
}

TEST_CASE("4x4 dense solve uses the upper triangle", "[solver]") {
  Mat4 matrix;
  matrix << 8.0, 1.0, 0.5, 0.25, 0.0, 7.0, 1.0, 0.5, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 0.0, 5.0;
  Mat4 symmetric;
  symmetric << 8.0, 1.0, 0.5, 0.25, 1.0, 7.0, 1.0, 0.5, 0.5, 1.0, 6.0, 1.0, 0.25, 0.5, 1.0, 5.0;
  const Vec4 expected(1.0, -2.0, 0.5, 3.0);
  const Vec4 rhs = symmetric * expected;
  DenseLinearSolvers<Mat4> solver_ldlt(LinearSolverMethod::LDLT);
  const auto solution = solver_ldlt.Solve(matrix, rhs);
  REQUIRE(solution.has_value());
  REQUIRE((*solution - expected).norm() < 1e-12);

  Eigen::Matrix<float, 4, 4> scaled = Eigen::Matrix<float, 4, 4>::Zero();
  scaled.diagonal() << 1.0f, 1.0f, 1.0f, 20000.0f;
  const Eigen::Matrix<float, 4, 1> scaled_expected(1.0f, -2.0f, 0.5f, 3.0f);
  DenseLinearSolvers<Eigen::Matrix<float, 4, 4>> solver_scaled(LinearSolverMethod::LDLT);
  const auto scaled_solution = solver_scaled.Solve(scaled, scaled * scaled_expected);
  REQUIRE(scaled_solution.has_value());
  REQUIRE((*scaled_solution - scaled_expected).norm() < 1e-5f);

  Mat4 singular = Mat4::Identity();
  singular(3, 3) = Eigen::NumTraits<double>::epsilon() * Eigen::NumTraits<double>::epsilon() * 0.5;
  REQUIRE_FALSE(solver_ldlt.Solve(singular, Vec4::Ones()).has_value());
}

TEST_CASE("invert_small_block", "[solver][invert]") {
  // 1x1
  Matrix<double, 1, 1> m1;
  m1 << 4.0;
  auto inv1 = Inverse(m1);
  REQUIRE(inv1.has_value());
  REQUIRE(Approx((*inv1)(0, 0)) == 0.25);

  // 2x2
  Mat2 m2;
  m2 << 4.0, 1.0, 1.0, 3.0;
  auto inv2 = Inverse(m2);
  REQUIRE(inv2.has_value());
  REQUIRE((m2 * (*inv2) - Mat2::Identity()).norm() < 1e-12);

  // 3x3
  Mat3 m3;
  m3 << 5.0, 1.0, 0.5, 1.0, 4.0, 0.2, 0.5, 0.2, 3.0;
  auto inv3 = Inverse(m3);
  REQUIRE(inv3.has_value());
  REQUIRE((m3 * (*inv3) - Mat3::Identity()).norm() < 1e-12);

  // 4x4
  Mat4 m4 = Mat4::Identity() * 3.0;
  m4(0, 1) = 0.5;
  m4(1, 0) = 0.5;
  auto inv4 = Inverse(m4);
  REQUIRE(inv4.has_value());
  REQUIRE((m4 * (*inv4) - Mat4::Identity()).norm() < 1e-12);

  // 5x5 (fallback to LDLT)
  Matrix<double, 5, 5> m5 = Matrix<double, 5, 5>::Identity() * 2.0;
  auto inv5 = Inverse(m5);
  REQUIRE(inv5.has_value());
  REQUIRE((m5 * (*inv5) - Matrix<double, 5, 5>::Identity()).norm() < 1e-12);

  // Singular matrix
  Mat3 m_sing = Mat3::Zero();
  REQUIRE_FALSE(Inverse(m_sing).has_value());
}

TEST_CASE("block_sparse_schur_top_left", "[solver][schur]") {
  // Generic 2-block system: 2 reduced blocks (size 6), 3 eliminated blocks (size 3) -> 12 + 9 = 21
  BlockSparseMat<6, 3> H(2, 3);
  REQUIRE(H.rows() == 21);

  // Set positive-definite diagonal blocks for reduced variables
  for (Index i = 0; i < 2; ++i) {
    Mat6 B = Mat6::Random();
    H.block<6, 6>(i, i) = B * B.transpose() + Mat6::Identity() * 10.0;
  }

  // Set positive-definite diagonal blocks for eliminated variables
  for (Index j = 0; j < 3; ++j) {
    Mat3 C = Mat3::Random();
    H.block<3, 3>(j, j) = C * C.transpose() + Mat3::Identity() * 5.0;
  }

  // Add coupling blocks
  Matrix<double, 6, 3> E00 = Matrix<double, 6, 3>::Random();
  Matrix<double, 6, 3> E01 = Matrix<double, 6, 3>::Random();
  Matrix<double, 6, 3> E11 = Matrix<double, 6, 3>::Random();
  Matrix<double, 6, 3> E12 = Matrix<double, 6, 3>::Random();

  H.block<6, 3>(0, 0) = E00;
  H.block<6, 3>(0, 1) = E01;
  H.block<6, 3>(1, 1) = E11;
  H.block<6, 3>(1, 2) = E12;

  // Ground truth solution x_expected
  VecX x_expected = VecX::Random(21);
  VecX b = H * x_expected;

  SECTION("natural ordering") {
    BlockSparseLinearSolvers<decltype(H)> solver;
    REQUIRE(solver.Analyze(H, 12));
    const auto solution = solver.Solve(H, b);
    REQUIRE(solution.has_value());
    REQUIRE((*solution - x_expected).norm() < 1e-10);
  }

  SECTION("full block-sparse system uses the direct linear solver") {
    BlockSparseLinearSolvers<decltype(H)> solver;
    REQUIRE(solver.Analyze(H));
    const auto solution = solver.Solve(H, b);
    REQUIRE(solution.has_value());
    REQUIRE((*solution - x_expected).norm() < 1e-10);

    H.block<6, 3>(0, 1).setZero();
    const VecX updated_rhs = H * x_expected;
    REQUIRE(solver.Compute(H));
    const auto updated_solution = solver.Solve(H, updated_rhs);
    REQUIRE(updated_solution.has_value());
    REQUIRE((*updated_solution - x_expected).norm() < 1e-10);
  }

  SECTION("Schur solving can be disabled") {
    LinearSolverOptions options{.allow_schur = false};
    BlockSparseLinearSolvers<decltype(H)> solver(options);
    REQUIRE_FALSE(solver.Analyze(H, 12));
    REQUIRE_FALSE(solver.Solve(H, b).has_value());
  }

  SECTION("singular eliminated diagonal block returns nullopt") {
    BlockSparseMat<6, 3> H_singular = H;
    H_singular.diagonal<3>(1).setZero();  // Eliminated block 1 is singular
    BlockSparseLinearSolvers<decltype(H_singular)> solver;
    REQUIRE(solver.Analyze(H_singular, 12));
    REQUIRE_FALSE(solver.Compute(H_singular));
    const auto solution = solver.Solve(H_singular, b);
    REQUIRE_FALSE(solution.has_value());
  }

  SECTION("non-finite rhs returns nullopt") {
    VecX b_nan = b;
    b_nan(0) = std::numeric_limits<double>::quiet_NaN();
    BlockSparseLinearSolvers<decltype(H)> solver;
    REQUIRE(solver.Analyze(H, 12));
    REQUIRE(solver.Compute(H));
    REQUIRE_FALSE(solver.Solve(H, b_nan).has_value());
  }
}

TEST_CASE("block sparse TopLeft Schur supports heterogeneous block sizes", "[solver][schur]") {
  using HeterogeneousMatrix = BlockSparseMatrix<float, 2, 3, 6>;
  HeterogeneousMatrix H(2, 1, 2);

  H.diagonal<2>(0) = Matrix<float, 2, 2>::Identity() * 8.0f;
  H.diagonal<2>(1) = Matrix<float, 2, 2>::Identity() * 9.0f;
  H.block<2, 2>(0, 1).setConstant(0.02f);
  H.diagonal<3>(0) = Matrix<float, 3, 3>::Identity() * 7.0f;
  H.diagonal<6>(0) = Matrix<float, 6, 6>::Identity() * 10.0f;
  H.diagonal<6>(1) = Matrix<float, 6, 6>::Identity() * 11.0f;

  H.block<2, 3>(0, 0).setConstant(0.03f);
  H.block<2, 6>(0, 0).setConstant(0.02f);
  H.block<2, 6>(1, 1).setConstant(0.025f);

  Vector<float, Dynamic> expected = Vector<float, Dynamic>::LinSpaced(H.rows(), -1.0f, 1.0f);
  const Vector<float, Dynamic> rhs = H * expected;

  SECTION("reduced corner ends before multiple eliminated block groups") {
    BlockSparseLinearSolvers<HeterogeneousMatrix> solver;
    REQUIRE(solver.Analyze(H, 4));
    REQUIRE(solver.Compute(H));
    const auto solution = solver.Solve(H, rhs);
    REQUIRE(solution.has_value());
    REQUIRE((*solution - expected).norm() < 1e-4f);
  }

  SECTION("reduced corner includes multiple block groups") {
    BlockSparseLinearSolvers<HeterogeneousMatrix> solver;
    REQUIRE(solver.Analyze(H, 7));
    REQUIRE(solver.Compute(H));
    const auto solution = solver.Solve(H, rhs);
    REQUIRE(solution.has_value());
    REQUIRE((*solution - expected).norm() < 1e-4f);
  }

  SECTION("rejects a reduced dimension inside a block-size group") {
    BlockSparseLinearSolvers<HeterogeneousMatrix> solver;
    REQUIRE_FALSE(solver.Analyze(H, 5));
    REQUIRE_FALSE(solver.Analyze(H, H.rows()));
  }

  SECTION("rejects off-diagonal blocks between eliminated variables") {
    HeterogeneousMatrix coupled_within_group = H;
    coupled_within_group.block<6, 6>(0, 1).setConstant(0.01f);
    BlockSparseLinearSolvers<HeterogeneousMatrix> same_group_solver;
    REQUIRE_FALSE(same_group_solver.Analyze(coupled_within_group, 4));

    HeterogeneousMatrix coupled_across_groups = H;
    coupled_across_groups.block<3, 6>(0, 1).setConstant(0.01f);
    BlockSparseLinearSolvers<HeterogeneousMatrix> cross_group_solver;
    REQUIRE_FALSE(cross_group_solver.Analyze(coupled_across_groups, 4));
  }

  SECTION("an eliminated group may couple to a reduced group") {
    HeterogeneousMatrix coupled = H;
    coupled.block<3, 6>(0, 0).setConstant(0.015f);
    coupled.block<3, 6>(0, 1).setConstant(0.01f);
    const Vector<float, Dynamic> coupled_rhs = coupled * expected;
    BlockSparseLinearSolvers<HeterogeneousMatrix> solver;
    REQUIRE(solver.Analyze(coupled, 7));
    const auto solution = solver.Solve(coupled, coupled_rhs);
    REQUIRE(solution.has_value());
    REQUIRE((*solution - expected).norm() < 1e-4f);
  }

  SECTION("reuses symbolic analysis when only block values change") {
    BlockSparseLinearSolvers<HeterogeneousMatrix> solver;
    REQUIRE(solver.Analyze(H, 4));
    REQUIRE(solver.Compute(H));
    REQUIRE(solver.IsAnalyzed());
    REQUIRE(solver.IsComputed());
    REQUIRE(solver.Solve(H, rhs).has_value());

    H.diagonal<6>(0)(0, 0) += 1.0f;
    const Vector<float, Dynamic> updated_rhs = H * expected;
    REQUIRE(solver.Compute(H));
    const auto updated_solution = solver.Solve(H, updated_rhs);
    REQUIRE(updated_solution.has_value());
    REQUIRE((*updated_solution - expected).norm() < 1e-4f);
  }

  SECTION("rejects a changed block pattern after analysis") {
    BlockSparseLinearSolvers<HeterogeneousMatrix> solver;
    REQUIRE(solver.Analyze(H, 4));
    H.block<2, 6>(0, 1).setZero();
    REQUIRE_FALSE(solver.Compute(H));
    const Vector<float, Dynamic> updated_rhs = H * expected;
    const auto solution = solver.Solve(H, updated_rhs);
    REQUIRE(solution.has_value());
    REQUIRE((*solution - expected).norm() < 1e-4f);
  }
}

TEST_CASE("sparse_schur_top_left", "[solver][schur]") {
  // Construct block-partitioned system in BlockSparseMat, then convert to SparseMat
  BlockSparseMat<6, 3> H_block(2, 3);
  for (Index i = 0; i < 2; ++i) {
    Mat6 B = Mat6::Random();
    H_block.block<6, 6>(i, i) = B * B.transpose() + Mat6::Identity() * 10.0;
  }
  for (Index j = 0; j < 3; ++j) {
    Mat3 C = Mat3::Random();
    H_block.block<3, 3>(j, j) = C * C.transpose() + Mat3::Identity() * 5.0;
  }
  H_block.block<6, 3>(0, 0) = Matrix<double, 6, 3>::Random();
  H_block.block<6, 3>(0, 1) = Matrix<double, 6, 3>::Random();
  H_block.block<6, 3>(1, 1) = Matrix<double, 6, 3>::Random();
  H_block.block<6, 3>(1, 2) = Matrix<double, 6, 3>::Random();

  SparseMat H_sparse = H_block.toSparse();
  VecX x_expected = VecX::Random(21);
  VecX b = H_sparse * x_expected;

  SECTION("natural ordering") {
    SparseLinearSolvers<SparseMat> solver;
    REQUIRE(solver.Analyze(H_sparse, 12));
    const auto solution = solver.Solve(H_sparse, b);
    REQUIRE(solution.has_value());
    REQUIRE((*solution - x_expected).norm() < 1e-9);
  }

  SECTION("dimension checks") {
    SparseLinearSolvers<SparseMat> solver;
    REQUIRE(solver.Analyze(H_sparse, 13));
    REQUIRE(solver.Solve(H_sparse, b).has_value());
    REQUIRE_FALSE(solver.Analyze(H_sparse, 21));
    REQUIRE_FALSE(solver.Analyze(H_sparse, -1));
  }
}
