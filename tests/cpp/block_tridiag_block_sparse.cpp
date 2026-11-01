// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/math/block_sparse_ldlt.h>
#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/math/block_tridiag_matrix.h>
#include <tinyopt/types.h>

using namespace tinyopt;
using Eigen::MatrixXd;

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
  REQUIRE((x_sol - x_expected).norm() < 1e-6);

  const auto opt_sol = ldlt.Solve(b);
  REQUIRE(opt_sol.has_value());
  REQUIRE((*opt_sol - x_expected).norm() < 1e-6);
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
  REQUIRE((x_sol - x_expected).norm() < 1e-6);

  const auto opt_sol = ldlt.Solve(b);
  REQUIRE(opt_sol.has_value());
  REQUIRE((*opt_sol - x_expected).norm() < 1e-6);
}
