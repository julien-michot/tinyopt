// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/math/block_sparse_ldlt.h>
#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/math/block_tridiagonal_ldlt.h>
#include <tinyopt/math/block_tridiagonal_matrix.h>
#include <tinyopt/optimize.h>
#include <tinyopt/types.h>

using namespace tinyopt;
using Eigen::MatrixXd;

TEST_CASE("BlockTridiagonalMatrix construction and dense conversion", "[tridiagonal][matrix]") {
  constexpr int B = 3;
  constexpr Index N = 5;
  BlockTridiagonalMat<B> T(N);

  REQUIRE(T.num_blocks() == N);
  REQUIRE(T.block_size() == B);
  REQUIRE(T.rows() == N * B);
  REQUIRE(T.cols() == N * B);

  // Fill diagonal and superdiagonal blocks
  for (Index i = 0; i < N; ++i) {
    Matrix<double, B, B> D = Matrix<double, B, B>::Random();
    T.diagonal(i) = D * D.transpose() + Matrix<double, B, B>::Identity() * 5.0;
  }
  for (Index i = 0; i < N - 1; ++i) {
    T.off_diagonal(i) = Matrix<double, B, B>::Random();
  }

  // Verify dense matrix symmetry and values
  const MatrixXd dense = T.toDense();
  REQUIRE(dense.rows() == N * B);
  REQUIRE(dense.cols() == N * B);
  REQUIRE((dense - dense.transpose()).norm() < 1e-12);

  // Check block access
  for (Index i = 0; i < N; ++i) {
    Matrix<double, B, B> d_dense = dense.block<B, B>(i * B, i * B);
    REQUIRE((d_dense - T.diagonal(i)).norm() < 1e-12);
  }
  for (Index i = 0; i < N - 1; ++i) {
    Matrix<double, B, B> u_dense = dense.block<B, B>(i * B, (i + 1) * B);
    REQUIRE((u_dense - T.off_diagonal(i)).norm() < 1e-12);
    Matrix<double, B, B> l_dense = dense.block<B, B>((i + 1) * B, i * B);
    REQUIRE((l_dense - T.off_diagonal(i).transpose()).norm() < 1e-12);
  }

  // Check matrix-vector product
  const VecX x = VecX::Random(N * B);
  const VecX y_tridiag = T * x;
  const VecX y_dense = dense * x;
  REQUIRE((y_tridiag - y_dense).norm() < 1e-12);

  // Check scaleDiagonal (Levenberg-Marquardt damping)
  T.scaleDiagonal(2.0);
  const MatrixXd damped_dense = T.toDense();
  for (Index i = 0; i < N; ++i) {
    for (int r = 0; r < B; ++r) {
      REQUIRE(Catch::Approx(damped_dense(i * B + r, i * B + r)) ==
              2.0 * dense(i * B + r, i * B + r));
    }
  }
}

using Case1 = std::integral_constant<int, 1>;
using Case2 = std::integral_constant<int, 2>;
using Case3 = std::integral_constant<int, 3>;
using Case4 = std::integral_constant<int, 4>;
using Case6 = std::integral_constant<int, 6>;
using CaseDyn = std::integral_constant<int, Dynamic>;

TEMPLATE_TEST_CASE("BlockTridiagonalLDLT vector and matrix solve against dense LDLT",
                   "[tridiagonal][ldlt]", Case1, Case2, Case3, Case4, Case6, CaseDyn) {
  constexpr int B = TestType::value;
  const int actual_B = (B == Dynamic) ? 5 : B;
  const Index N = 6;
  const Index dims = N * actual_B;

  BlockTridiagonalMatrix<double, B> T(N, actual_B);

  for (Index i = 0; i < N; ++i) {
    MatrixXd D = MatrixXd::Random(actual_B, actual_B);
    T.diagonal(i) = D * D.transpose() + MatrixXd::Identity(actual_B, actual_B) * 10.0;
  }
  for (Index i = 0; i < N - 1; ++i) {
    T.off_diagonal(i) = MatrixXd::Random(actual_B, actual_B) * 0.5;
  }

  const MatrixXd dense = T.toDense();
  const Eigen::LDLT<MatrixXd> dense_ldlt(dense);
  REQUIRE(dense_ldlt.info() == Eigen::Success);

  BlockTridiagonalLDLT<double, B> tridiag_ldlt;
  tridiag_ldlt.analyzePattern(T);
  REQUIRE(tridiag_ldlt.isAnalyzed());

  tridiag_ldlt.compute(T);
  REQUIRE(tridiag_ldlt.isComputed());
  REQUIRE(tridiag_ldlt.info() == Eigen::Success);

  SECTION("Vector RHS solve") {
    const VecX b = VecX::Random(dims);
    const VecX x_dense = dense_ldlt.solve(b);
    const VecX x_tridiag = tridiag_ldlt.solve(b);

    REQUIRE((x_tridiag - x_dense).norm() < 1e-10);

    const auto opt_sol = tridiag_ldlt.Solve(b);
    REQUIRE(opt_sol.has_value());
    REQUIRE((*opt_sol - x_dense).norm() < 1e-10);
  }

  SECTION("Matrix RHS solve (Inverse recovery)") {
    const MatrixXd I = MatrixXd::Identity(dims, dims);
    const MatrixXd inv_dense = dense_ldlt.solve(I);
    const MatrixXd inv_tridiag = tridiag_ldlt.solve(I);

    REQUIRE((inv_tridiag - inv_dense).norm() < 1e-10);
    REQUIRE((dense * inv_tridiag - I).norm() < 1e-10);
  }

  SECTION("Zero-copy computeFromPointers") {
    std::vector<const typename BlockTridiagonalLDLT<double, B>::Block*> diag_ptrs(
        static_cast<size_t>(N));
    std::vector<const typename BlockTridiagonalLDLT<double, B>::Block*> off_diag_ptrs(
        static_cast<size_t>(N - 1));

    for (Index i = 0; i < N; ++i) {
      diag_ptrs[static_cast<size_t>(i)] = &T.diagonal(i);
    }
    for (Index i = 0; i < N - 1; ++i) {
      off_diag_ptrs[static_cast<size_t>(i)] = &T.off_diagonal(i);
    }

    BlockTridiagonalLDLT<double, B> ptr_ldlt;
    ptr_ldlt.analyzePattern(N, actual_B);
    ptr_ldlt.computeFromPointers(diag_ptrs, off_diag_ptrs);
    REQUIRE(ptr_ldlt.isComputed());
    REQUIRE(ptr_ldlt.info() == Eigen::Success);

    const VecX b = VecX::Random(dims);
    const VecX x_ptr = ptr_ldlt.solve(b);
    const VecX x_dense = dense_ldlt.solve(b);
    REQUIRE((x_ptr - x_dense).norm() < 1e-10);
  }
}

TEST_CASE("BlockSparseLDLT auto-detection of full tridiagonal system",
          "[block_sparse][tridiagonal]") {
  constexpr int B = 6;
  constexpr Index N = 5;
  const Index dims = N * B;

  BlockSparseMat<B> H(N);

  for (Index i = 0; i < N; ++i) {
    Matrix<double, B, B> D = Matrix<double, B, B>::Random();
    H.diagonal<B>(i) = D * D.transpose() + Matrix<double, B, B>::Identity() * 10.0;
  }
  for (Index i = 0; i < N - 1; ++i) {
    H.block<B, B>(i, i + 1) = Matrix<double, B, B>::Random() * 0.2;
  }

  BlockSparseLDLT<decltype(H), BlockNaturalOrdering> ldlt;
  ldlt.analyzePattern(H, 0);

  REQUIRE(ldlt.isAnalyzed());
  REQUIRE(ldlt.isFullTridiagonal());

  ldlt.compute(H);
  REQUIRE(ldlt.isComputed());
  REQUIRE(ldlt.info() == Eigen::Success);

  const VecX x_expected = VecX::Random(dims);
  const VecX b = H * x_expected;

  const VecX x_sol = ldlt.solve(b);
  REQUIRE((x_sol - x_expected).norm() < 1e-10);

  const auto opt_sol = ldlt.Solve(b);
  REQUIRE(opt_sol.has_value());
  REQUIRE((*opt_sol - x_expected).norm() < 1e-10);
}

TEST_CASE("BlockSparseLDLT Tier-2 Schur elimination with tridiagonal eliminated chain",
          "[block_sparse][schur][tridiagonal]") {
  // 1 reduced block of size 6 (e.g. keyframe or sensor base)
  // 4 eliminated states of size 3 (e.g. trajectory points coupled in time)
  BlockSparseMat<6, 3> H(1, 4);

  // Reduced diagonal
  Matrix<double, 6, 6> R = Matrix<double, 6, 6>::Random();
  H.diagonal<6>(0) = R * R.transpose() + Matrix<double, 6, 6>::Identity() * 20.0;

  // Eliminated tridiagonal chain
  for (Index j = 0; j < 4; ++j) {
    Matrix<double, 3, 3> E = Matrix<double, 3, 3>::Random();
    H.diagonal<3>(j) = E * E.transpose() + Matrix<double, 3, 3>::Identity() * 10.0;
  }
  for (Index j = 0; j < 3; ++j) {
    H.block<3, 3>(j, j + 1) = Matrix<double, 3, 3>::Random() * 0.3;
  }

  // Coupling between reduced block and eliminated states
  H.block<6, 3>(0, 0) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(0, 2) = Matrix<double, 6, 3>::Random();

  const Index dims = 6 + 4 * 3;  // 18
  const VecX x_expected = VecX::Random(dims);
  const VecX b = H * x_expected;

  BlockSparseLDLT<decltype(H), BlockNaturalOrdering> ldlt;
  ldlt.analyzePattern(H, 6);

  REQUIRE(ldlt.isAnalyzed());
  REQUIRE(ldlt.numReduced() == 6);
  REQUIRE(ldlt.numEliminated() == 12);
  REQUIRE(ldlt.isEliminatedTridiagonal());

  ldlt.compute(H);
  REQUIRE(ldlt.isComputed());
  REQUIRE(ldlt.info() == Eigen::Success);

  const VecX x_sol = ldlt.solve(b);
  REQUIRE((x_sol - x_expected).norm() < 1e-10);

  const auto opt_sol = ldlt.Solve(b);
  REQUIRE(opt_sol.has_value());
  REQUIRE((*opt_sol - x_expected).norm() < 1e-10);
}

TEST_CASE("Optimizer with BlockTridiagonalMat Hessian", "[tridiagonal][optimizer]") {
  // State: 4 nodes of dimension 2 (Vec8)
  using Hessian = BlockTridiagonalMat<2>;
  using Vec8 = Vector<double, 8>;
  Vec8 x;
  x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;

  const Vec8 target = Vec8::Constant(2.0);

  auto loss = [&](const Vec8& val, Vector<double, Dynamic>& grad, Hessian& H) {
    H.setZero();
    grad.setZero();
    double total_cost = 0.0;

    // Node target priors: 0.5 * ||x_i - target_i||^2
    for (Index i = 0; i < 4; ++i) {
      Vec2 diff = val.segment<2>(i * 2) - target.segment<2>(i * 2);
      total_cost += 0.5 * diff.squaredNorm();
      grad.segment<2>(i * 2) += diff;
      H.diagonal(i).noalias() += Matrix<double, 2, 2>::Identity();
    }

    // Edge smoothness coupling: 0.5 * w * ||x_{i+1} - x_i||^2
    const double w = 2.0;
    for (Index i = 0; i < 3; ++i) {
      Vec2 diff = val.segment<2>((i + 1) * 2) - val.segment<2>(i * 2);
      total_cost += 0.5 * w * diff.squaredNorm();
      grad.segment<2>(i * 2) -= w * diff;
      grad.segment<2>((i + 1) * 2) += w * diff;

      H.diagonal(i).noalias() += w * Matrix<double, 2, 2>::Identity();
      H.diagonal(i + 1).noalias() += w * Matrix<double, 2, 2>::Identity();
      H.off_diagonal(i).noalias() -= w * Matrix<double, 2, 2>::Identity();
    }

    return total_cost;
  };

  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.damping_init = 1e-4;
  options.stop.max_iters = 20;

  const auto summary = Optimize(x, loss, options);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE((x - target).norm() < 1e-6);
}
